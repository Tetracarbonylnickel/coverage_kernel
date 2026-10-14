!==================================================================================================================================
!> Reads the HDF5 schema (docs/hdf5_schema.md) written by
!> coverage_kernel/python/coverage_kernel/export_hdf5.py into a
!> t_CoverageModel. Every shape quantity (n_classes, n_terms, grid shape,
!> knot-vector length, layer widths, number of members) is discovered from
!> the file itself, never assumed.
!>
!> 0-based -> 1-based index conversion (outcome-class indices, term raw-
!> feature columns, gate indices) happens exactly here.
!>
!> Every dataset with more than one dimension is read as a flat 1-D buffer
!> and indexed with the C-order (last-axis-fastest) convention of
!> MOD_CoverageKernel_Interp1D's FlatIndexCOrder -- never into a native
!> multi-dimensional Fortran array, which would silently transpose it.
!>
!> HDF5 library state: h5open_f is called (it is safe to call repeatedly),
!> h5close_f is never called, so a host code that uses HDF5 itself (e.g.
!> PICLas) keeps a working library. Every HDF5 call is checked; a file that
!> exists but does not match the schema stops with a message naming the
!> failing item.
!==================================================================================================================================
MODULE MOD_CoverageKernel_IO
! MODULES
USE HDF5
USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_NAN
USE MOD_CoverageKernel_Types,    ONLY: wp, NAME_LEN, t_Term, t_SplineClassifier, t_GridAxis, t_GridCell, &
                                        t_GridModel, t_NNModel, t_MLP, t_CoverageModel
USE MOD_CoverageKernel_MLP,      ONLY: IsKnownActivation
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
PUBLIC :: ReadCoverageModelHDF5

!> Wraps one flat REAL buffer (one spline term's whole term_knots/<i> dataset).
TYPE :: t_FlatBuf
  REAL(wp), ALLOCATABLE :: Data(:)
END TYPE t_FlatBuf
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> Top-level reader. IErr = 0 on success, 1 if the file does not exist,
!> 2 if it exists but HDF5 cannot open it. Everything after a successful
!> open is a schema check that stops with a message on failure (a model file
!> that opens but does not match the schema is a deployment error, not a
!> runtime condition).
!==================================================================================================================================
SUBROUTINE ReadCoverageModelHDF5(FileName, Model, IErr)
! INPUT / OUTPUT VARIABLES
CHARACTER(LEN=*),      INTENT(IN)  :: FileName
TYPE(t_CoverageModel), INTENT(OUT) :: Model
INTEGER,               INTENT(OUT) :: IErr
! LOCAL VARIABLES
INTEGER(HID_T) :: FileID, TermsGroupID
INTEGER        :: hdferr
LOGICAL        :: Exists
TYPE(t_Term), ALLOCATABLE :: Terms(:)
!----------------------------------------------------------------------------------------------------------------------------
INQUIRE(FILE=TRIM(FileName), EXIST=Exists)
IF (.NOT. Exists) THEN
  IErr = 1
  RETURN
END IF
CALL h5open_f(hdferr)
CALL Chk(hdferr, "h5open_f", "")
CALL h5fopen_f(TRIM(FileName), H5F_ACC_RDONLY_F, FileID, hdferr)
IF (hdferr < 0) THEN
  IErr = 2
  RETURN
END IF
IErr = 0

Model%FormatVersion = H5ReadStrAttr(FileID, "format_version")
IF (TRIM(Model%FormatVersion) /= "1.0" .AND. TRIM(Model%FormatVersion) /= "1.1") THEN
  WRITE(*,*) "MOD_CoverageKernel_IO: ", TRIM(FileName), " has format_version '", TRIM(Model%FormatVersion), &
             "', this reader supports 1.0 and 1.1"
  ERROR STOP "MOD_CoverageKernel_IO: unsupported format_version"
END IF
Model%Scheme             = H5ReadStrAttr(FileID, "scheme")
Model%NSites             = H5ReadIntAttr(FileID, "n_sites")
Model%ZeroedZeroIndex    = H5ReadIntAttr(FileID, "zeroed_zero_index")    + 1
Model%ZeroedFullIndex    = H5ReadIntAttr(FileID, "zeroed_full_index")    + 1
Model%AbsorbingFullIndex = H5ReadIntAttr(FileID, "absorbing_full_index") + 1
Model%NClasses           = H5ReadIntAttr(FileID, "n_classes")
Model%CoverageDelta      = H5ReadRealFlat(FileID, "coverage_delta")
IF (SIZE(Model%CoverageDelta) /= Model%NClasses) ERROR STOP "MOD_CoverageKernel_IO: coverage_delta length /= n_classes"

Model%HasModelArea = H5AttrExists(FileID, "model_area_nm2")
Model%ModelAreaNm2 = 0.0_wp
IF (Model%HasModelArea) Model%ModelAreaNm2 = H5ReadRealAttr(FileID, "model_area_nm2")

SELECT CASE (TRIM(Model%Scheme))
CASE ("grid")
  CALL h5gopen_f(FileID, "terms", TermsGroupID, hdferr)
  CALL Chk(hdferr, "open group", "terms")
  CALL ReadTermSpec(TermsGroupID, Terms)
  CALL h5gclose_f(TermsGroupID, hdferr)
  CALL ReadGridModel(FileID, Terms, Model%NClasses, Model%Grid)
CASE ("nn_ensemble")
  CALL ReadNNModel(FileID, Model%NClasses, Model%NN)
CASE DEFAULT
  WRITE(*,*) "MOD_CoverageKernel_IO: ", TRIM(FileName), " has scheme '", TRIM(Model%Scheme), "'"
  ERROR STOP "MOD_CoverageKernel_IO: unknown /scheme (expected 'grid' or 'nn_ensemble')"
END SELECT

CALL h5fclose_f(FileID, hdferr)
CALL Chk(hdferr, "close file", FileName)
END SUBROUTINE ReadCoverageModelHDF5

!==================================================================================================================================
!> Stop with a message if an HDF5 call failed (HDF5 Fortran returns a
!> negative hdferr on failure).
!==================================================================================================================================
SUBROUTINE Chk(hdferr, What, Name)
INTEGER,          INTENT(IN) :: hdferr
CHARACTER(LEN=*), INTENT(IN) :: What, Name
IF (hdferr < 0) THEN
  WRITE(*,*) "MOD_CoverageKernel_IO: HDF5 call failed: ", TRIM(What), " ", TRIM(Name)
  ERROR STOP "MOD_CoverageKernel_IO: model file does not match the schema (see message above)"
END IF
END SUBROUTINE Chk

!==================================================================================================================================
!> Strip a fixed-width HDF5 string at its first NUL byte.
!==================================================================================================================================
PURE FUNCTION CStrClean(S) RESULT(Out)
CHARACTER(LEN=*), INTENT(IN) :: S
CHARACTER(LEN=LEN(S)) :: Out
INTEGER :: p
p = INDEX(S, ACHAR(0))
IF (p > 0) THEN
  Out = S(1:p-1)
ELSE
  Out = S
END IF
END FUNCTION CStrClean

!==================================================================================================================================
!> Integer as a decimal string without blanks (names like "term_knots/3",
!> "member_0", "layer_2" follow 0-based Python indices).
!==================================================================================================================================
PURE FUNCTION IntToStr(N) RESULT(S)
INTEGER, INTENT(IN) :: N
CHARACTER(LEN=:), ALLOCATABLE :: S
CHARACTER(LEN=16) :: Buf
WRITE(Buf, '(I0)') N
S = TRIM(Buf)
END FUNCTION IntToStr

!==================================================================================================================================
!> Open a child group, stopping with its name if it does not exist.
!==================================================================================================================================
SUBROUTINE OpenGroup(LocID, Name, GroupID)
INTEGER(HID_T),   INTENT(IN)  :: LocID
CHARACTER(LEN=*), INTENT(IN)  :: Name
INTEGER(HID_T),   INTENT(OUT) :: GroupID
INTEGER :: hdferr
CALL h5gopen_f(LocID, TRIM(Name), GroupID, hdferr)
CALL Chk(hdferr, "open group", Name)
END SUBROUTINE OpenGroup

SUBROUTINE CloseGroup(GroupID)
INTEGER(HID_T), INTENT(IN) :: GroupID
INTEGER :: hdferr
CALL h5gclose_f(GroupID, hdferr)
CALL Chk(hdferr, "close group", "")
END SUBROUTINE CloseGroup

!==================================================================================================================================
!> HDF5 accessor helpers: scalar int/real/string attributes, attribute
!> existence, and flat (1-D, any original rank) dataset reads.
!==================================================================================================================================
FUNCTION H5AttrExists(LocID, Name) RESULT(Exists)
INTEGER(HID_T),   INTENT(IN) :: LocID
CHARACTER(LEN=*), INTENT(IN) :: Name
LOGICAL :: Exists
INTEGER :: hdferr
CALL h5aexists_f(LocID, TRIM(Name), Exists, hdferr)
CALL Chk(hdferr, "check attribute", Name)
END FUNCTION H5AttrExists

FUNCTION H5ReadIntAttr(LocID, Name) RESULT(Val)
INTEGER(HID_T),   INTENT(IN) :: LocID
CHARACTER(LEN=*), INTENT(IN) :: Name
INTEGER :: Val
INTEGER(HID_T)   :: AttrID
INTEGER(HSIZE_T) :: Dims(1)
INTEGER :: hdferr
Dims(1) = 1
CALL h5aopen_f(LocID, TRIM(Name), AttrID, hdferr)
CALL Chk(hdferr, "open attribute", Name)
CALL h5aread_f(AttrID, H5T_NATIVE_INTEGER, Val, Dims, hdferr)
CALL Chk(hdferr, "read attribute", Name)
CALL h5aclose_f(AttrID, hdferr)
END FUNCTION H5ReadIntAttr

FUNCTION H5ReadRealAttr(LocID, Name) RESULT(Val)
INTEGER(HID_T),   INTENT(IN) :: LocID
CHARACTER(LEN=*), INTENT(IN) :: Name
REAL(wp) :: Val
INTEGER(HID_T)   :: AttrID
INTEGER(HSIZE_T) :: Dims(1)
INTEGER :: hdferr
Dims(1) = 1
CALL h5aopen_f(LocID, TRIM(Name), AttrID, hdferr)
CALL Chk(hdferr, "open attribute", Name)
CALL h5aread_f(AttrID, H5T_NATIVE_DOUBLE, Val, Dims, hdferr)
CALL Chk(hdferr, "read attribute", Name)
CALL h5aclose_f(AttrID, hdferr)
END FUNCTION H5ReadRealAttr

FUNCTION H5ReadStrAttr(LocID, Name) RESULT(Str)
INTEGER(HID_T),   INTENT(IN) :: LocID
CHARACTER(LEN=*), INTENT(IN) :: Name
CHARACTER(LEN=NAME_LEN) :: Str
INTEGER(HID_T)   :: AttrID, StrType
INTEGER(HSIZE_T) :: Dims(1)
INTEGER :: hdferr
Dims(1) = 1
Str = ' '
CALL h5tcopy_f(H5T_FORTRAN_S1, StrType, hdferr)
CALL h5tset_size_f(StrType, INT(NAME_LEN, SIZE_T), hdferr)
CALL h5aopen_f(LocID, TRIM(Name), AttrID, hdferr)
CALL Chk(hdferr, "open attribute", Name)
CALL h5aread_f(AttrID, StrType, Str, Dims, hdferr)
CALL Chk(hdferr, "read attribute", Name)
CALL h5aclose_f(AttrID, hdferr)
CALL h5tclose_f(StrType, hdferr)
Str = CStrClean(Str)
END FUNCTION H5ReadStrAttr

FUNCTION H5DatasetTotalSize(LocID, Name) RESULT(N)
INTEGER(HID_T),   INTENT(IN) :: LocID
CHARACTER(LEN=*), INTENT(IN) :: Name
INTEGER :: N
INTEGER(HID_T) :: DsetID, SpaceID
INTEGER :: hdferr, Rank
INTEGER(HSIZE_T), ALLOCATABLE :: HDims(:), HMaxDims(:)
CALL h5dopen_f(LocID, TRIM(Name), DsetID, hdferr)
CALL Chk(hdferr, "open dataset", Name)
CALL h5dget_space_f(DsetID, SpaceID, hdferr)
CALL h5sget_simple_extent_ndims_f(SpaceID, Rank, hdferr)
CALL Chk(hdferr, "get rank of dataset", Name)
ALLOCATE(HDims(Rank), HMaxDims(Rank))
CALL h5sget_simple_extent_dims_f(SpaceID, HDims, HMaxDims, hdferr)
CALL Chk(hdferr, "get dimensions of dataset", Name)
N = INT(PRODUCT(HDims))
CALL h5sclose_f(SpaceID, hdferr)
CALL h5dclose_f(DsetID, hdferr)
END FUNCTION H5DatasetTotalSize

FUNCTION H5ReadRealFlat(LocID, Name) RESULT(Buf)
INTEGER(HID_T),   INTENT(IN) :: LocID
CHARACTER(LEN=*), INTENT(IN) :: Name
REAL(wp), ALLOCATABLE :: Buf(:)
INTEGER(HID_T)   :: DsetID
INTEGER(HSIZE_T) :: Dims(1)
INTEGER :: hdferr, N
N = H5DatasetTotalSize(LocID, Name)
ALLOCATE(Buf(N))
Dims(1) = N
CALL h5dopen_f(LocID, TRIM(Name), DsetID, hdferr)
CALL Chk(hdferr, "open dataset", Name)
CALL h5dread_f(DsetID, H5T_NATIVE_DOUBLE, Buf, Dims, hdferr)
CALL Chk(hdferr, "read dataset", Name)
CALL h5dclose_f(DsetID, hdferr)
END FUNCTION H5ReadRealFlat

FUNCTION H5ReadIntFlat(LocID, Name) RESULT(Buf)
INTEGER(HID_T),   INTENT(IN) :: LocID
CHARACTER(LEN=*), INTENT(IN) :: Name
INTEGER, ALLOCATABLE :: Buf(:)
INTEGER(HID_T)   :: DsetID
INTEGER(HSIZE_T) :: Dims(1)
INTEGER :: hdferr, N
N = H5DatasetTotalSize(LocID, Name)
ALLOCATE(Buf(N))
Dims(1) = N
CALL h5dopen_f(LocID, TRIM(Name), DsetID, hdferr)
CALL Chk(hdferr, "open dataset", Name)
CALL h5dread_f(DsetID, H5T_NATIVE_INTEGER, Buf, Dims, hdferr)
CALL Chk(hdferr, "read dataset", Name)
CALL h5dclose_f(DsetID, hdferr)
END FUNCTION H5ReadIntFlat

FUNCTION H5ReadStrArray(LocID, Name, N) RESULT(Buf)
INTEGER(HID_T),   INTENT(IN) :: LocID
CHARACTER(LEN=*), INTENT(IN) :: Name
INTEGER,          INTENT(IN) :: N
CHARACTER(LEN=NAME_LEN), ALLOCATABLE :: Buf(:)
INTEGER(HID_T)   :: DsetID, StrType
INTEGER(HSIZE_T) :: Dims(1)
INTEGER :: hdferr, i
ALLOCATE(Buf(N))
Dims(1) = N
CALL h5tcopy_f(H5T_FORTRAN_S1, StrType, hdferr)
CALL h5tset_size_f(StrType, INT(NAME_LEN, SIZE_T), hdferr)
CALL h5dopen_f(LocID, TRIM(Name), DsetID, hdferr)
CALL Chk(hdferr, "open dataset", Name)
CALL h5dread_f(DsetID, StrType, Buf, Dims, hdferr)
CALL Chk(hdferr, "read dataset", Name)
CALL h5dclose_f(DsetID, hdferr)
CALL h5tclose_f(StrType, hdferr)
DO i = 1, N
  Buf(i) = CStrClean(Buf(i))
END DO
END FUNCTION H5ReadStrArray

!> Number of immediate child members of a group (the number of ceiling axes
!> is the number of subgroups, as in the Python loader).
FUNCTION H5NumGroupMembers(GroupID) RESULT(NMembers)
INTEGER(HID_T), INTENT(IN) :: GroupID
INTEGER :: NMembers
INTEGER :: hdferr, StorageType, MaxCOrder
CALL h5gget_info_f(GroupID, StorageType, NMembers, MaxCOrder, hdferr)
CALL Chk(hdferr, "count group members", "")
END FUNCTION H5NumGroupMembers

!==================================================================================================================================
!> Read the shared term spec (/terms): one t_Term per fitted term,
!> NFeaturesOut computed from NKnots/Degree as spline_classifier.py's
!> _n_term_features does. Knot values are per cell (read in ReadGridModel).
!==================================================================================================================================
SUBROUTINE ReadTermSpec(GroupID, Terms)
INTEGER(HID_T),            INTENT(IN)  :: GroupID
TYPE(t_Term), ALLOCATABLE, INTENT(OUT) :: Terms(:)
INTEGER :: NTerms, t
INTEGER, ALLOCATABLE :: ColumnBuf(:), DegreeBuf(:), NKnotsBuf(:)
CHARACTER(LEN=NAME_LEN), ALLOCATABLE :: KindBuf(:)
!----------------------------------------------------------------------------------------------------------------------------
NTerms = H5ReadIntAttr(GroupID, "n_terms")
ALLOCATE(Terms(NTerms))
ColumnBuf = H5ReadIntFlat(GroupID, "column")
DegreeBuf = H5ReadIntFlat(GroupID, "degree")
NKnotsBuf = H5ReadIntFlat(GroupID, "n_knots")
KindBuf   = H5ReadStrArray(GroupID, "kind", NTerms)

DO t = 1, NTerms
  Terms(t)%Kind   = TRIM(KindBuf(t))
  Terms(t)%Column = ColumnBuf(t) + 1   ! 0-based -> 1-based
  IF (TRIM(Terms(t)%Kind) == "spline") THEN
    Terms(t)%Degree = DegreeBuf(t)
    Terms(t)%NKnots = NKnotsBuf(t)
    Terms(t)%NFeaturesOut = Terms(t)%NKnots + Terms(t)%Degree - 2
  ELSE IF (TRIM(Terms(t)%Kind) == "linear") THEN
    Terms(t)%Degree       = -1
    Terms(t)%NKnots       = -1
    Terms(t)%NFeaturesOut = 1
  ELSE
    ERROR STOP "MOD_CoverageKernel_IO: unknown term kind (expected 'spline' or 'linear')"
  END IF
END DO
END SUBROUTINE ReadTermSpec

!==================================================================================================================================
!> Read a ceiling grid (ceiling_axes/<d>/values + ceiling_grid) below GroupID.
!> The ceiling is always interpolated multilinearly and clipped to its axes.
!==================================================================================================================================
SUBROUTINE ReadCeilingGrid(GroupID, Axes, Shape_, Grid)
INTEGER(HID_T),                 INTENT(IN)  :: GroupID
TYPE(t_GridAxis), ALLOCATABLE,  INTENT(OUT) :: Axes(:)
INTEGER, ALLOCATABLE,           INTENT(OUT) :: Shape_(:)
REAL(wp), ALLOCATABLE,          INTENT(OUT) :: Grid(:)
INTEGER(HID_T) :: CeilAxesGroupID, SubGroupID
INTEGER :: d, NAxes
!----------------------------------------------------------------------------------------------------------------------------
CALL OpenGroup(GroupID, "ceiling_axes", CeilAxesGroupID)
NAxes = H5NumGroupMembers(CeilAxesGroupID)
ALLOCATE(Axes(NAxes), Shape_(NAxes))
DO d = 1, NAxes
  CALL OpenGroup(CeilAxesGroupID, IntToStr(d - 1), SubGroupID)
  Axes(d)%Values     = H5ReadRealFlat(SubGroupID, "values")
  Axes(d)%EdgePolicy = "clip"
  Shape_(d)          = SIZE(Axes(d)%Values)
  CALL CloseGroup(SubGroupID)
END DO
CALL CloseGroup(CeilAxesGroupID)
Grid = H5ReadRealFlat(GroupID, "ceiling_grid")
IF (SIZE(Grid) /= PRODUCT(Shape_)) ERROR STOP "MOD_CoverageKernel_IO: ceiling_grid size does not match its axes"
END SUBROUTINE ReadCeilingGrid

!==================================================================================================================================
!> Read the "nn_ensemble" scheme (/nn): every member network with its input
!> standardization, the offset class, and the ceiling grid. Mirrors
!> export_hdf5.py's load_nn_model. Weights are stored (n_in, n_out) in
!> C-order; that flat buffer reshaped to the Fortran shape (NOut, NIn) gives
!> W(j, i) = weights[i][j], so Y = MATMUL(W, X) + B.
!==================================================================================================================================
SUBROUTINE ReadNNModel(FileID, NClasses, NN)
INTEGER(HID_T),  INTENT(IN)  :: FileID
INTEGER,         INTENT(IN)  :: NClasses
TYPE(t_NNModel), INTENT(OUT) :: NN
INTEGER(HID_T) :: GroupID, MemberID, LayerID
INTEGER :: m, l, NIn, NOut
REAL(wp), ALLOCATABLE :: WFlat(:)
!----------------------------------------------------------------------------------------------------------------------------
CALL OpenGroup(FileID, "nn", GroupID)
IF (TRIM(H5ReadStrAttr(GroupID, "combination")) /= "mean") &
  ERROR STOP "MOD_CoverageKernel_IO: /nn/combination must be 'mean'"
NN%NMembers    = H5ReadIntAttr(GroupID, "n_members")
NN%CoverageMax = H5ReadRealAttr(GroupID, "coverage_max")
NN%OffsetClass = H5ReadIntAttr(GroupID, "offset_class") + 1
NN%OffsetFloor = H5ReadRealAttr(GroupID, "offset_floor")
NN%HasCeiling  = H5ReadIntAttr(GroupID, "has_ceiling") /= 0
ALLOCATE(NN%Members(NN%NMembers))

DO m = 1, NN%NMembers
  CALL OpenGroup(GroupID, "member_" // IntToStr(m - 1), MemberID)
  ASSOCIATE(Net => NN%Members(m))
    Net%InputMean = H5ReadRealFlat(MemberID, "input_mean")
    Net%InputStd  = H5ReadRealFlat(MemberID, "input_std")
    Net%NLayers   = H5ReadIntAttr(MemberID, "n_layers")
    ALLOCATE(Net%Layers(Net%NLayers))
    NIn = SIZE(Net%InputMean) + 1          ! conditioning inputs + coverage fraction
    DO l = 1, Net%NLayers
      CALL OpenGroup(MemberID, "layer_" // IntToStr(l - 1), LayerID)
      Net%Layers(l)%Activation = H5ReadStrAttr(LayerID, "activation")
      IF (.NOT. IsKnownActivation(Net%Layers(l)%Activation)) &
        ERROR STOP "MOD_CoverageKernel_IO: unknown activation in /nn (expected silu, tanh, relu or linear)"
      Net%Layers(l)%B = H5ReadRealFlat(LayerID, "biases")
      WFlat = H5ReadRealFlat(LayerID, "weights")
      NOut = SIZE(Net%Layers(l)%B)
      IF (SIZE(WFlat) /= NIn * NOut) ERROR STOP "MOD_CoverageKernel_IO: /nn layer weights do not match the layer widths"
      Net%Layers(l)%NIn  = NIn
      Net%Layers(l)%NOut = NOut
      Net%Layers(l)%W = RESHAPE(WFlat, [NOut, NIn])
      CALL CloseGroup(LayerID)
      NIn = NOut
    END DO
    IF (NIn /= NClasses) ERROR STOP "MOD_CoverageKernel_IO: last /nn layer width /= n_classes"
  END ASSOCIATE
  CALL CloseGroup(MemberID)
END DO

IF (NN%HasCeiling) THEN
  CALL ReadCeilingGrid(GroupID, NN%CeilingAxes, NN%CeilingShape, NN%CeilingGrid)
  NN%NCeilingAxes = SIZE(NN%CeilingAxes)
ELSE
  NN%NCeilingAxes = 0
END IF
CALL CloseGroup(GroupID)
END SUBROUTINE ReadNNModel

!==================================================================================================================================
!> Read the "grid" scheme (/grid): the axes (with per-axis edge policy) and
!> every cell, flattened C-order. Mirrors export_hdf5.py's load_grid_model.
!==================================================================================================================================
SUBROUTINE ReadGridModel(FileID, Terms, NClasses, Model)
INTEGER(HID_T),    INTENT(IN)  :: FileID
TYPE(t_Term),      INTENT(IN)  :: Terms(:)
INTEGER,           INTENT(IN)  :: NClasses
TYPE(t_GridModel), INTENT(OUT) :: Model
INTEGER(HID_T) :: GroupID, AxesGroupID, SubGroupID, CellsGroupID, TermKnotsGroupID
INTEGER :: d, NAxes, NFeatures, NCells, c, t, p
INTEGER :: NPresentC, BaseFeat, BasePres, BaseCoef, BaseInt, BaseKnot, KnotLen
REAL(wp) :: CeilVal
REAL(wp), ALLOCATABLE :: CoverageMaxFlat(:), CeilingFlat(:), ScalerMeanFlat(:), ScalerScaleFlat(:)
REAL(wp), ALLOCATABLE :: CoefFlat(:), InterceptFlat(:)
INTEGER,  ALLOCATABLE :: NPresentFlat(:), PresentFlat(:)
TYPE(t_FlatBuf), ALLOCATABLE :: TermKnotFlats(:)
!----------------------------------------------------------------------------------------------------------------------------
CALL OpenGroup(FileID, "grid", GroupID)

NAxes = H5ReadIntAttr(GroupID, "n_axes")
Model%NAxes = NAxes
Model%InterpMethod = H5ReadStrAttr(GroupID, "interp_method")
ALLOCATE(Model%Axes(NAxes), Model%Shape(NAxes))

CALL OpenGroup(GroupID, "axes", AxesGroupID)
DO d = 1, NAxes
  CALL OpenGroup(AxesGroupID, IntToStr(d - 1), SubGroupID)
  Model%Axes(d)%Values     = H5ReadRealFlat(SubGroupID, "values")
  Model%Axes(d)%EdgePolicy = H5ReadStrAttr(SubGroupID, "edge_policy")
  Model%Shape(d)           = SIZE(Model%Axes(d)%Values)
  CALL CloseGroup(SubGroupID)
END DO
CALL CloseGroup(AxesGroupID)

NFeatures = SUM(Terms(:)%NFeaturesOut)
NCells = PRODUCT(Model%Shape)

CALL OpenGroup(GroupID, "cells", CellsGroupID)
CoverageMaxFlat = H5ReadRealFlat(CellsGroupID, "coverage_max")
CeilingFlat     = H5ReadRealFlat(CellsGroupID, "ceiling")
ScalerMeanFlat  = H5ReadRealFlat(CellsGroupID, "scaler_mean")
ScalerScaleFlat = H5ReadRealFlat(CellsGroupID, "scaler_scale")
NPresentFlat    = H5ReadIntFlat(CellsGroupID, "n_present_classes")
PresentFlat     = H5ReadIntFlat(CellsGroupID, "present_classes")
CoefFlat        = H5ReadRealFlat(CellsGroupID, "coef")
InterceptFlat   = H5ReadRealFlat(CellsGroupID, "intercept")
IF (SIZE(CoverageMaxFlat) /= NCells .OR. SIZE(CoefFlat) /= NCells * NClasses * NFeatures) &
  ERROR STOP "MOD_CoverageKernel_IO: /grid/cells sizes do not match the axes and terms"

ALLOCATE(Model%Cells(NCells))

CALL OpenGroup(CellsGroupID, "term_knots", TermKnotsGroupID)
ALLOCATE(TermKnotFlats(SIZE(Terms)))
DO t = 1, SIZE(Terms)
  IF (TRIM(Terms(t)%Kind) == "spline") THEN
    TermKnotFlats(t)%Data = H5ReadRealFlat(TermKnotsGroupID, IntToStr(t - 1))
  END IF
END DO

DO c = 1, NCells
  ASSOCIATE(Cell => Model%Cells(c))
    Cell%CoverageMax = CoverageMaxFlat(c)
    CeilVal = CeilingFlat(c)
    Cell%HasCeiling = .NOT. IEEE_IS_NAN(CeilVal)
    Cell%Ceiling = MERGE(CeilVal, 0.0_wp, Cell%HasCeiling)

    Cell%Classifier%NTerms    = SIZE(Terms)
    Cell%Classifier%NFeatures = NFeatures
    Cell%Classifier%NClasses  = NClasses
    ALLOCATE(Cell%Classifier%Terms(SIZE(Terms)))
    Cell%Classifier%Terms = Terms

    BaseFeat = (c - 1) * NFeatures
    Cell%Classifier%ScalerMean  = ScalerMeanFlat(BaseFeat + 1 : BaseFeat + NFeatures)
    Cell%Classifier%ScalerScale = ScalerScaleFlat(BaseFeat + 1 : BaseFeat + NFeatures)

    NPresentC = NPresentFlat(c)
    Cell%Classifier%NPresentClasses = NPresentC
    BasePres = (c - 1) * NClasses
    ALLOCATE(Cell%Classifier%PresentClasses(NPresentC))
    Cell%Classifier%PresentClasses = PresentFlat(BasePres + 1 : BasePres + NPresentC) + 1

    BaseCoef = (c - 1) * NClasses * NFeatures
    BaseInt  = (c - 1) * NClasses
    ALLOCATE(Cell%Classifier%Coef(NPresentC, NFeatures), Cell%Classifier%Intercept(NPresentC))
    DO p = 1, NPresentC
      Cell%Classifier%Coef(p, :) = CoefFlat(BaseCoef + (p - 1) * NFeatures + 1 : BaseCoef + p * NFeatures)
      Cell%Classifier%Intercept(p) = InterceptFlat(BaseInt + p)
    END DO

    DO t = 1, SIZE(Terms)
      IF (TRIM(Terms(t)%Kind) == "spline") THEN
        KnotLen  = Terms(t)%NKnots + 2 * Terms(t)%Degree
        BaseKnot = (c - 1) * KnotLen
        Cell%Classifier%Terms(t)%Knots = TermKnotFlats(t)%Data(BaseKnot + 1 : BaseKnot + KnotLen)
      END IF
    END DO
  END ASSOCIATE
END DO

CALL CloseGroup(TermKnotsGroupID)
CALL CloseGroup(CellsGroupID)
CALL CloseGroup(GroupID)
END SUBROUTINE ReadGridModel

END MODULE MOD_CoverageKernel_IO
