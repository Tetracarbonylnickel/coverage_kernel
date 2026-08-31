!==================================================================================================================================
!> Reads the HDF5 schema (docs/hdf5_schema.md) written by
!> coverage_kernel/python/coverage_kernel/export_hdf5.py into a
!> t_CoverageModel. Every shape quantity (n_classes, n_terms, grid shape,
!> knot-vector length, ...) is discovered from the file itself via HDF5
!> dataspace/attribute queries, never assumed.
!>
!> 0-based -> 1-based index conversion (outcome-class indices, term raw-
!> feature columns, gate indices) happens exactly here, nowhere else in
!> the Fortran port -- everything past this module operates on 1-based
!> Fortran indices only.
!>
!> Every dataset with more than one leading grid/cell dimension is read as
!> a flat 1-D buffer and indexed with the same C-order (last-axis-fastest)
!> convention MOD_CoverageKernel_Interp1D's FlatIndexCOrder and
!> MOD_CoverageKernel_Grid use -- never into a native multi-dimensional
!> Fortran array. See docs/hdf5_schema.md's notes section for why: Fortran
!> arrays are column-major but every array this schema stores is written
!> in C-order (NumPy's default), so a direct multi-dim h5dread_f would
!> silently transpose the data relative to what indexing Buf(i0+1,i1+1,...)
!> should mean.
!==================================================================================================================================
MODULE MOD_CoverageKernel_IO
! MODULES
USE HDF5
USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_NAN
USE MOD_CoverageKernel_Types,    ONLY: wp, NAME_LEN, t_Term, t_PooledClassifier, t_GridAxis, t_GridCell, &
                                        t_PooledModel, t_GridModel, t_CoverageModel
USE MOD_CoverageKernel_Interp1D, ONLY: FlatIndexCOrder
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
PUBLIC :: ReadCoverageModelHDF5

!> Wraps one flat REAL buffer -- used to hold one spline term's whole
!> term_knots/<i> dataset (read once) while ReadGridModel slices per-cell
!> knot vectors out of it in memory, since SIZE(Terms) buffers of
!> different lengths can't be a plain rank-2 array.
TYPE :: t_FlatBuf
  REAL(wp), ALLOCATABLE :: Data(:)
END TYPE t_FlatBuf
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> Top-level HDF5 reader: opens FileName, dispatches to the pooled or grid
!> reader based on the /scheme attribute, populates every field of Model.
!> IErr is 0 on success, nonzero if the file could not be opened (the one
!> failure mode a caller can reasonably expect to happen at runtime, e.g.
!> a bad path). A malformed/incompatible file that opens fine but doesn't
!> match this schema is treated as a programmer/build error, not a
!> recoverable runtime condition, and still hits ERROR STOP directly
!> inside the lower-level readers below -- see MOD_CoverageKernel's
!> LoadCoverageModel for how IErr is surfaced (or not) to callers.
!==================================================================================================================================
SUBROUTINE ReadCoverageModelHDF5(FileName, Model, IErr)
! INPUT / OUTPUT VARIABLES
CHARACTER(LEN=*),      INTENT(IN)  :: FileName
TYPE(t_CoverageModel), INTENT(OUT) :: Model
INTEGER,               INTENT(OUT) :: IErr
! LOCAL VARIABLES
INTEGER(HID_T) :: FileID, TermsGroupID
INTEGER        :: hdferr
TYPE(t_Term), ALLOCATABLE :: Terms(:)
!----------------------------------------------------------------------------------------------------------------------------
CALL h5open_f(hdferr)
CALL h5fopen_f(TRIM(FileName), H5F_ACC_RDONLY_F, FileID, hdferr)
IF (hdferr /= 0) THEN
  IErr = 1
  RETURN
END IF
IErr = 0

Model%Scheme = H5ReadStrAttr(FileID, "scheme")
Model%NSites = H5ReadIntAttr(FileID, "n_sites")
Model%ZeroedZeroIndex    = H5ReadIntAttr(FileID, "zeroed_zero_index")    + 1
Model%ZeroedFullIndex    = H5ReadIntAttr(FileID, "zeroed_full_index")    + 1
Model%AbsorbingFullIndex = H5ReadIntAttr(FileID, "absorbing_full_index") + 1
Model%NClasses = H5ReadIntAttr(FileID, "n_classes")
Model%CoverageDelta = H5ReadRealFlat(FileID, "coverage_delta")

Model%HasModelArea = H5AttrExists(FileID, "model_area_nm2")
IF (Model%HasModelArea) THEN
  Model%ModelAreaNm2 = H5ReadRealAttr(FileID, "model_area_nm2")
ELSE
  Model%ModelAreaNm2 = 0.0_wp
END IF

CALL h5gopen_f(FileID, "terms", TermsGroupID, hdferr)
CALL ReadTermSpec(TermsGroupID, Terms)
CALL h5gclose_f(TermsGroupID, hdferr)

SELECT CASE (TRIM(Model%Scheme))
CASE ("pooled")
  CALL ReadPooledModel(FileID, Terms, Model%NClasses, Model%Pooled)
CASE ("grid")
  CALL ReadGridModel(FileID, Terms, Model%NClasses, Model%Grid)
CASE DEFAULT
  ERROR STOP "MOD_CoverageKernel_IO: unknown /scheme (expected 'pooled' or 'grid')"
END SELECT

CALL h5fclose_f(FileID, hdferr)
CALL h5close_f(hdferr)
END SUBROUTINE ReadCoverageModelHDF5

!==================================================================================================================================
!> Strip a fixed-width HDF5 string buffer at its first NUL byte (HDF5
!> null-pads fixed-length strings shorter than the declared width) and
!> trim trailing spaces.
!==================================================================================================================================
PURE FUNCTION CStrClean(S) RESULT(Out)
! INPUT / OUTPUT VARIABLES
CHARACTER(LEN=*), INTENT(IN) :: S
CHARACTER(LEN=LEN(S)) :: Out
! LOCAL VARIABLES
INTEGER :: p
!----------------------------------------------------------------------------------------------------------------------------
p = INDEX(S, ACHAR(0))
IF (p > 0) THEN
  Out = S(1:p-1)
ELSE
  Out = S
END IF
END FUNCTION CStrClean

!==================================================================================================================================
!> Format an integer as a plain decimal string with no leading blanks --
!> used to build HDF5 child-group/dataset names like "term_knots/3" or
!> "ceiling_axes/0" that export_hdf5.py names after a 0-based Python index.
!==================================================================================================================================
PURE FUNCTION IntToStr(N) RESULT(S)
! INPUT / OUTPUT VARIABLES
INTEGER, INTENT(IN) :: N
CHARACTER(LEN=:), ALLOCATABLE :: S
! LOCAL VARIABLES
CHARACTER(LEN=16) :: Buf
!----------------------------------------------------------------------------------------------------------------------------
WRITE(Buf, '(I0)') N
S = TRIM(Buf)
END FUNCTION IntToStr

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
CALL h5aread_f(AttrID, H5T_NATIVE_INTEGER, Val, Dims, hdferr)
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
CALL h5aread_f(AttrID, H5T_NATIVE_DOUBLE, Val, Dims, hdferr)
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
CALL h5aread_f(AttrID, StrType, Str, Dims, hdferr)
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
CALL h5dget_space_f(DsetID, SpaceID, hdferr)
CALL h5sget_simple_extent_ndims_f(SpaceID, Rank, hdferr)
ALLOCATE(HDims(Rank), HMaxDims(Rank))
CALL h5sget_simple_extent_dims_f(SpaceID, HDims, HMaxDims, hdferr)
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
CALL h5dread_f(DsetID, H5T_NATIVE_DOUBLE, Buf, Dims, hdferr)
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
CALL h5dread_f(DsetID, H5T_NATIVE_INTEGER, Buf, Dims, hdferr)
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
CALL h5dread_f(DsetID, StrType, Buf, Dims, hdferr)
CALL h5dclose_f(DsetID, hdferr)
CALL h5tclose_f(StrType, hdferr)
DO i = 1, N
  Buf(i) = CStrClean(Buf(i))
END DO
END FUNCTION H5ReadStrArray

!> Number of immediate child members of a group (used where the schema
!> doesn't store an explicit count, e.g. /pooled/ceiling_axes -- the
!> number of conditioning dimensions is just "however many subgroups are
!> there", mirroring the Python loader's len(ceiling_axes_group)).
FUNCTION H5NumGroupMembers(GroupID) RESULT(NMembers)
INTEGER(HID_T), INTENT(IN) :: GroupID
INTEGER :: NMembers
INTEGER :: hdferr, StorageType, MaxCOrder
CALL h5gget_info_f(GroupID, StorageType, NMembers, MaxCOrder, hdferr)
END FUNCTION H5NumGroupMembers

!==================================================================================================================================
!> Read the shared term spec (/terms in the schema): one t_Term per fitted
!> term, NFeaturesOut computed from NKnots/Degree exactly as
!> pooled_classifier.py's _n_term_features does. Per-term fitted Knots
!> values are read later (they live under whichever of /pooled or
!> /grid/cells actually holds the fitted classifier(s), since a spline
!> term's knot *values* are fit data, not part of the shared spec).
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
  ELSE
    Terms(t)%Degree       = -1
    Terms(t)%NKnots       = -1
    Terms(t)%NFeaturesOut = 1
  END IF
END DO
END SUBROUTINE ReadTermSpec

!==================================================================================================================================
!> Read one fitted t_PooledClassifier from GroupID (either /pooled itself,
!> for the pooled scheme, or -- via ReadGridCells -- effectively inlined
!> per grid cell for the grid scheme). SharedTerms carries the Kind/
!> Column/Degree/NKnots/NFeaturesOut fields already read by ReadTermSpec;
!> only the per-fit Knots values, present-class bookkeeping, and
!> coefficients are read here.
!==================================================================================================================================
FUNCTION ReadSinglePooledClassifier(GroupID, SharedTerms, NClasses) RESULT(Clf)
INTEGER(HID_T),   INTENT(IN) :: GroupID
TYPE(t_Term),     INTENT(IN) :: SharedTerms(:)
INTEGER,          INTENT(IN) :: NClasses
TYPE(t_PooledClassifier) :: Clf
INTEGER :: NFeatures, NPresent, t, t0, p
INTEGER(HID_T) :: TermKnotsGroupID
INTEGER :: hdferr
INTEGER,  ALLOCATABLE :: PresentBuf(:)
REAL(wp), ALLOCATABLE :: CoefFlat(:), InterceptBuf(:)
!----------------------------------------------------------------------------------------------------------------------------
NFeatures = SUM(SharedTerms(:)%NFeaturesOut)

Clf%NTerms    = SIZE(SharedTerms)
Clf%NFeatures = NFeatures
Clf%NClasses  = NClasses
ALLOCATE(Clf%Terms(SIZE(SharedTerms)))
Clf%Terms = SharedTerms

Clf%ScalerMean  = H5ReadRealFlat(GroupID, "scaler_mean")
Clf%ScalerScale = H5ReadRealFlat(GroupID, "scaler_scale")

NPresent = H5ReadIntAttr(GroupID, "n_present_classes")
Clf%NPresentClasses = NPresent
PresentBuf = H5ReadIntFlat(GroupID, "present_classes")
ALLOCATE(Clf%PresentClasses(NPresent))
Clf%PresentClasses = PresentBuf(1:NPresent) + 1   ! 0-based -> 1-based

CoefFlat     = H5ReadRealFlat(GroupID, "coef")        ! (NClasses, NFeatures), C-order; first NPresent rows valid
InterceptBuf = H5ReadRealFlat(GroupID, "intercept")   ! (NClasses,)

ALLOCATE(Clf%Coef(NPresent, NFeatures), Clf%Intercept(NPresent))
DO p = 1, NPresent
  Clf%Coef(p, :)  = CoefFlat((p - 1) * NFeatures + 1 : p * NFeatures)
  Clf%Intercept(p) = InterceptBuf(p)
END DO

CALL h5gopen_f(GroupID, "term_knots", TermKnotsGroupID, hdferr)
DO t = 1, SIZE(SharedTerms)
  IF (TRIM(SharedTerms(t)%Kind) == "spline") THEN
    t0 = t - 1
    Clf%Terms(t)%Knots = H5ReadRealFlat(TermKnotsGroupID, IntToStr(t0))
  END IF
END DO
CALL h5gclose_f(TermKnotsGroupID, hdferr)
END FUNCTION ReadSinglePooledClassifier

!==================================================================================================================================
!> Read the "pooled" scheme's whole domain model: the single fitted
!> classifier plus its optional interpolated ceiling grid. Mirrors
!> export_hdf5.py's load_pooled_model.
!==================================================================================================================================
SUBROUTINE ReadPooledModel(FileID, Terms, NClasses, Model)
INTEGER(HID_T),        INTENT(IN)  :: FileID
TYPE(t_Term),          INTENT(IN)  :: Terms(:)
INTEGER,               INTENT(IN)  :: NClasses
TYPE(t_PooledModel),   INTENT(OUT) :: Model
INTEGER(HID_T) :: GroupID, CeilAxesGroupID, SubGroupID
INTEGER :: hdferr, d, NCeilAxes
!----------------------------------------------------------------------------------------------------------------------------
CALL h5gopen_f(FileID, "pooled", GroupID, hdferr)

Model%Classifier  = ReadSinglePooledClassifier(GroupID, Terms, NClasses)
Model%CoverageMax = H5ReadRealAttr(GroupID, "coverage_max")
Model%HasCeiling  = H5ReadIntAttr(GroupID, "has_ceiling") /= 0

IF (Model%HasCeiling) THEN
  CALL h5gopen_f(GroupID, "ceiling_axes", CeilAxesGroupID, hdferr)
  NCeilAxes = H5NumGroupMembers(CeilAxesGroupID)
  Model%NCeilingAxes = NCeilAxes
  ALLOCATE(Model%CeilingAxes(NCeilAxes), Model%CeilingShape(NCeilAxes))
  DO d = 1, NCeilAxes
    CALL h5gopen_f(CeilAxesGroupID, IntToStr(d - 1), SubGroupID, hdferr)
    Model%CeilingAxes(d)%Values     = H5ReadRealFlat(SubGroupID, "values")
    Model%CeilingAxes(d)%EdgePolicy = "clip"   ! ceiling interpolation always clips (see MultilinearClipped)
    Model%CeilingShape(d)           = SIZE(Model%CeilingAxes(d)%Values)
    CALL h5gclose_f(SubGroupID, hdferr)
  END DO
  CALL h5gclose_f(CeilAxesGroupID, hdferr)
  Model%CeilingGrid = H5ReadRealFlat(GroupID, "ceiling_grid")
ELSE
  Model%NCeilingAxes = 0
END IF

CALL h5gclose_f(GroupID, hdferr)
END SUBROUTINE ReadPooledModel

!==================================================================================================================================
!> Read the "grid" scheme's whole domain model: the axes (with per-axis
!> edge policy) and every already-blended cell, flattened C-order.
!> Mirrors export_hdf5.py's load_grid_model.
!==================================================================================================================================
SUBROUTINE ReadGridModel(FileID, Terms, NClasses, Model)
INTEGER(HID_T),    INTENT(IN)  :: FileID
TYPE(t_Term),      INTENT(IN)  :: Terms(:)
INTEGER,           INTENT(IN)  :: NClasses
TYPE(t_GridModel), INTENT(OUT) :: Model
INTEGER(HID_T) :: GroupID, AxesGroupID, SubGroupID, CellsGroupID, TermKnotsGroupID
INTEGER :: hdferr, d, NAxes, NFeatures, NCells, c, t, t0, p
INTEGER :: NPresentC, BaseFeat, BasePres, BaseCoef, BaseInt, BaseKnot, KnotLen
REAL(wp) :: CeilVal
REAL(wp), ALLOCATABLE :: CoverageMaxFlat(:), CeilingFlat(:), ScalerMeanFlat(:), ScalerScaleFlat(:)
REAL(wp), ALLOCATABLE :: CoefFlat(:), InterceptFlat(:)
INTEGER,  ALLOCATABLE :: NPresentFlat(:), PresentFlat(:)
TYPE(t_FlatBuf), ALLOCATABLE :: TermKnotFlats(:)
!----------------------------------------------------------------------------------------------------------------------------
CALL h5gopen_f(FileID, "grid", GroupID, hdferr)

NAxes = H5ReadIntAttr(GroupID, "n_axes")
Model%NAxes = NAxes
Model%InterpMethod = H5ReadStrAttr(GroupID, "interp_method")
ALLOCATE(Model%Axes(NAxes), Model%Shape(NAxes))

CALL h5gopen_f(GroupID, "axes", AxesGroupID, hdferr)
DO d = 1, NAxes
  CALL h5gopen_f(AxesGroupID, IntToStr(d - 1), SubGroupID, hdferr)
  Model%Axes(d)%Values     = H5ReadRealFlat(SubGroupID, "values")
  Model%Axes(d)%EdgePolicy = H5ReadStrAttr(SubGroupID, "edge_policy")
  Model%Shape(d)           = SIZE(Model%Axes(d)%Values)
  CALL h5gclose_f(SubGroupID, hdferr)
END DO
CALL h5gclose_f(AxesGroupID, hdferr)

NFeatures = SUM(Terms(:)%NFeaturesOut)
NCells = PRODUCT(Model%Shape)

CALL h5gopen_f(GroupID, "cells", CellsGroupID, hdferr)
CoverageMaxFlat = H5ReadRealFlat(CellsGroupID, "coverage_max")
CeilingFlat     = H5ReadRealFlat(CellsGroupID, "ceiling")
ScalerMeanFlat  = H5ReadRealFlat(CellsGroupID, "scaler_mean")
ScalerScaleFlat = H5ReadRealFlat(CellsGroupID, "scaler_scale")
NPresentFlat    = H5ReadIntFlat(CellsGroupID, "n_present_classes")
PresentFlat     = H5ReadIntFlat(CellsGroupID, "present_classes")
CoefFlat        = H5ReadRealFlat(CellsGroupID, "coef")
InterceptFlat   = H5ReadRealFlat(CellsGroupID, "intercept")

ALLOCATE(Model%Cells(NCells))

CALL h5gopen_f(CellsGroupID, "term_knots", TermKnotsGroupID, hdferr)
ALLOCATE(TermKnotFlats(SIZE(Terms)))
DO t = 1, SIZE(Terms)
  IF (TRIM(Terms(t)%Kind) == "spline") THEN
    t0 = t - 1
    TermKnotFlats(t)%Data = H5ReadRealFlat(TermKnotsGroupID, IntToStr(t0))
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

CALL h5gclose_f(TermKnotsGroupID, hdferr)
CALL h5gclose_f(CellsGroupID, hdferr)
CALL h5gclose_f(GroupID, hdferr)
END SUBROUTINE ReadGridModel

END MODULE MOD_CoverageKernel_IO
