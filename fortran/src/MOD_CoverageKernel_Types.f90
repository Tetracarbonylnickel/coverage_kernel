!==================================================================================================================================
!> Shared derived types for the coverage_kernel Fortran port.
!>
!> Every shape quantity (number of classes, number of terms, number of grid
!> axes, axis lengths, spline knot counts) is a runtime property of the
!> loaded model -- read from the HDF5 file at MOD_CoverageKernel_IO's
!> LoadCoverageModel, never a compiled-in constant. The two exceptions are
!> genuine properties of the chosen interpolation methods rather than of any
!> particular dataset: MAX_WINDOW (the largest local interpolation stencil
!> any supported 1-D method ever needs) and the string-length parameters
!> used for HDF5-stored method/policy names.
!==================================================================================================================================
MODULE MOD_CoverageKernel_Types
! MODULES
USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: REAL64
IMPLICIT NONE
PUBLIC
SAVE
!----------------------------------------------------------------------------------------------------------------------------
INTEGER, PARAMETER :: wp = REAL64                 !> working precision, explicit rather than bare REAL
INTEGER, PARAMETER :: MAX_WINDOW = 4               !> largest local 1-D stencil (pchip: up to 4 points); a property
                                                    !> of the interpolation method, not of any dataset's shape
INTEGER, PARAMETER :: NAME_LEN = 32                 !> string length for method/policy/kind tags read from HDF5

!==================================================================================================================================
!> One term of a pooled classifier's feature expansion: either a B-spline
!> basis expansion of one raw-feature column ("spline"), or that column
!> passed through unchanged ("linear"). Mirrors pooled_classifier.py's
!> SplineTerm/LinearTerm.
!==================================================================================================================================
TYPE :: t_Term
  CHARACTER(LEN=NAME_LEN)   :: Kind               !> "spline" or "linear"
  INTEGER                   :: Column              !> 1-based column index into the raw feature vector
  INTEGER                   :: Degree              !> spline terms only; -1 for linear terms
  INTEGER                   :: NKnots              !> spline terms only (fit parameter, not raw knot count); -1 for linear
  INTEGER                   :: NFeaturesOut        !> width this term contributes to the assembled feature vector
  REAL(wp), ALLOCATABLE     :: Knots(:)            !> fitted extended knot vector (spline terms only)
END TYPE t_Term

!==================================================================================================================================
!> A fitted K-class softmax-logistic classifier over a feature vector built
!> from Terms(:). Mirrors pooled_classifier.py's PooledClassifier: classes
!> never observed at fit time are absent from Coef/Intercept entirely
!> (PresentClasses lists which of the NClasses possible outcomes were
!> actually fit) rather than stored as a zero-coefficient row, so evaluation
!> must scatter the softmax result into the full NClasses-length output
!> exactly as EvalPooledClassifier does -- never read Coef as if it were
!> already (NClasses, NFeatures).
!==================================================================================================================================
TYPE :: t_PooledClassifier
  INTEGER                    :: NTerms
  TYPE(t_Term), ALLOCATABLE  :: Terms(:)
  INTEGER                    :: NFeatures
  REAL(wp), ALLOCATABLE      :: ScalerMean(:)      !> (NFeatures)
  REAL(wp), ALLOCATABLE      :: ScalerScale(:)     !> (NFeatures)
  INTEGER                    :: NClasses           !> total possible classes K (includes never-observed ones)
  INTEGER                    :: NPresentClasses
  INTEGER, ALLOCATABLE       :: PresentClasses(:)  !> 1-based class indices, length NPresentClasses
  REAL(wp), ALLOCATABLE      :: Coef(:,:)          !> (NPresentClasses, NFeatures)
  REAL(wp), ALLOCATABLE      :: Intercept(:)       !> (NPresentClasses)
END TYPE t_PooledClassifier

!==================================================================================================================================
!> One axis of a grid model (e.g. velocity or angle): its sampled values
!> (sorted) and its out-of-range query policy. Mirrors grid_classifier.py's
!> AxisSpec.
!==================================================================================================================================
TYPE :: t_GridAxis
  REAL(wp), ALLOCATABLE      :: Values(:)
  CHARACTER(LEN=NAME_LEN)    :: EdgePolicy         !> "clip" or "linear_extrapolate"
END TYPE t_GridAxis

!==================================================================================================================================
!> One grid cell of the "grid" scheme: an independent classifier plus the
!> coverage blend/gate physics already resolved for that cell (its own
!> CoverageMax/Ceiling -- see coverage_model.py's CoverageCell). Blending is
!> done per cell before cross-cell interpolation, so every cell needs its
!> own copy of the physics gate indices even though those are the same
!> across the whole model; storing them per cell keeps the grid evaluator
!> (MOD_CoverageKernel_Grid) able to evaluate a cell without a second
!> pointer back into the parent model.
!==================================================================================================================================
TYPE :: t_GridCell
  TYPE(t_PooledClassifier)   :: Classifier
  REAL(wp)                   :: CoverageMax
  REAL(wp)                   :: Ceiling
  LOGICAL                    :: HasCeiling
END TYPE t_GridCell

!==================================================================================================================================
!> Domain wrapper for the "pooled" scheme: one shared classifier over
!> engineered features, plus an optional ceiling grid interpolated (always
!> multilinear, never pchip -- it's a smooth safety boundary, not a fitted
!> probability curve) across whatever conditioning axes it was computed on.
!> Mirrors coverage_model.py's PooledCoverageModel.
!==================================================================================================================================
TYPE :: t_PooledModel
  TYPE(t_PooledClassifier)      :: Classifier
  REAL(wp)                       :: CoverageMax
  LOGICAL                        :: HasCeiling
  INTEGER                        :: NCeilingAxes
  TYPE(t_GridAxis), ALLOCATABLE  :: CeilingAxes(:)
  INTEGER, ALLOCATABLE           :: CeilingShape(:)   !> (NCeilingAxes)
  REAL(wp), ALLOCATABLE          :: CeilingGrid(:)     !> flattened, C-order (last axis fastest); see FlatIndex
END TYPE t_PooledModel

!==================================================================================================================================
!> Domain wrapper for the "grid" scheme: an N-axis grid of independent,
!> already-blended classifiers (Cells, flattened C-order -- last axis
!> fastest, matching how MOD_CoverageKernel_IO reads the HDF5 arrays and how
!> MOD_CoverageKernel_Grid's FlatIndex computes offsets). Mirrors
!> grid_classifier.py's GridClassifier (with coverage_model.py's
!> CoverageCell wrapping already applied to every cell at fit/export time).
!==================================================================================================================================
TYPE :: t_GridModel
  INTEGER                        :: NAxes
  TYPE(t_GridAxis), ALLOCATABLE  :: Axes(:)
  INTEGER, ALLOCATABLE           :: Shape(:)          !> (NAxes)
  CHARACTER(LEN=NAME_LEN)        :: InterpMethod       !> default cross-cell method: "pchip" or "linear"
  TYPE(t_GridCell), ALLOCATABLE  :: Cells(:)           !> flattened, length = PRODUCT(Shape)
END TYPE t_GridModel

!==================================================================================================================================
!> Top-level loaded model: whichever of t_PooledModel/t_GridModel was
!> stored, plus the shared coverage physics/metadata every scheme needs.
!> Mirrors the union of coverage_model.py's CoveragePhysics with whichever
!> concrete model class export_hdf5.py wrote.
!==================================================================================================================================
TYPE :: t_CoverageModel
  CHARACTER(LEN=NAME_LEN)  :: Scheme                !> "pooled" or "grid"
  INTEGER                  :: NClasses
  INTEGER                  :: NSites
  INTEGER                  :: ZeroedZeroIndex        !> 1-based class index impossible at coverage==0
  INTEGER                  :: ZeroedFullIndex        !> 1-based class index impossible at coverage>=NSites
  INTEGER                  :: AbsorbingFullIndex     !> 1-based class absorbing ZeroedFullIndex's freed mass
  REAL(wp), ALLOCATABLE    :: CoverageDelta(:)       !> (NClasses)
  LOGICAL                  :: HasModelArea
  REAL(wp)                 :: ModelAreaNm2
  TYPE(t_PooledModel)      :: Pooled                 !> valid iff Scheme == "pooled"
  TYPE(t_GridModel)        :: Grid                   !> valid iff Scheme == "grid"
END TYPE t_CoverageModel

END MODULE MOD_CoverageKernel_Types
