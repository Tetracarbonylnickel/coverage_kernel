!==================================================================================================================================
!> Shared derived types for the coverage_kernel Fortran port.
!>
!> Every shape quantity (number of classes, number of terms, number of grid
!> axes, axis lengths, spline knot counts, network layer widths, number of
!> ensemble members) is a runtime property of the loaded model -- read from
!> the HDF5 file by MOD_CoverageKernel_IO, never a compiled-in constant. The
!> exceptions are properties of the interpolation methods rather than of any
!> dataset: MAX_WINDOW (the largest local interpolation stencil any supported
!> 1-D method needs) and the length of the string tags stored in the file.
!==================================================================================================================================
MODULE MOD_CoverageKernel_Types
! MODULES
USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: REAL64
IMPLICIT NONE
PUBLIC
SAVE
!----------------------------------------------------------------------------------------------------------------------------
INTEGER, PARAMETER :: wp = REAL64                 !> working precision
INTEGER, PARAMETER :: MAX_WINDOW = 4               !> largest local 1-D stencil (pchip: up to 4 points)
INTEGER, PARAMETER :: NAME_LEN = 32                 !> length of the string tags read from HDF5

!==================================================================================================================================
!> One term of a spline classifier's feature expansion: either a B-spline
!> basis expansion of one raw-feature column ("spline"), or that column
!> passed through unchanged ("linear"). Mirrors spline_classifier.py's
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
!> from Terms(:). Mirrors spline_classifier.py's SplineClassifier: classes
!> never observed at fit time are absent from Coef/Intercept entirely
!> (PresentClasses lists which of the NClasses possible outcomes were
!> actually fit), so evaluation scatters the softmax result into the full
!> NClasses-length output exactly as EvalSplineClassifier does.
!==================================================================================================================================
TYPE :: t_SplineClassifier
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
END TYPE t_SplineClassifier

!==================================================================================================================================
!> One axis of a grid (e.g. velocity or angle): its sampled values (sorted)
!> and its out-of-range query policy. Mirrors grid_classifier.py's AxisSpec.
!==================================================================================================================================
TYPE :: t_GridAxis
  REAL(wp), ALLOCATABLE      :: Values(:)
  CHARACTER(LEN=NAME_LEN)    :: EdgePolicy         !> "clip" or "linear_extrapolate"
END TYPE t_GridAxis

!==================================================================================================================================
!> One cell of the "grid" scheme: an independent classifier plus the
!> coverage blend limits of that cell (its own CoverageMax/Ceiling -- see
!> coverage_model.py's CoverageCell). Blending is done per cell before the
!> interpolation across cells.
!==================================================================================================================================
TYPE :: t_GridCell
  TYPE(t_SplineClassifier)   :: Classifier
  REAL(wp)                   :: CoverageMax
  REAL(wp)                   :: Ceiling
  LOGICAL                    :: HasCeiling
END TYPE t_GridCell

!==================================================================================================================================
!> The "grid" scheme: an N-axis grid of independent classifiers (Cells,
!> flattened C-order, last axis fastest). Mirrors grid_classifier.py's
!> GridClassifier with coverage_model.py's CoverageCell around every cell.
!==================================================================================================================================
TYPE :: t_GridModel
  INTEGER                        :: NAxes
  TYPE(t_GridAxis), ALLOCATABLE  :: Axes(:)
  INTEGER, ALLOCATABLE           :: Shape(:)          !> (NAxes)
  CHARACTER(LEN=NAME_LEN)        :: InterpMethod       !> cross-cell method: "pchip" or "linear"
  TYPE(t_GridCell), ALLOCATABLE  :: Cells(:)           !> flattened, length = PRODUCT(Shape)
END TYPE t_GridModel

!==================================================================================================================================
!> One dense layer of a feed-forward network: Y = Act(W X + B).
!> W is stored as (NOut, NIn); the file stores (n_in, n_out) in C-order,
!> which is the same flat buffer (see MOD_CoverageKernel_IO).
!==================================================================================================================================
TYPE :: t_DenseLayer
  INTEGER                    :: NIn, NOut
  REAL(wp), ALLOCATABLE      :: W(:,:)             !> (NOut, NIn)
  REAL(wp), ALLOCATABLE      :: B(:)               !> (NOut)
  CHARACTER(LEN=NAME_LEN)    :: Activation         !> "silu", "tanh", "relu" or "linear"
END TYPE t_DenseLayer

!==================================================================================================================================
!> One network of the ensemble, with the standardization of its
!> conditioning inputs (velocity, angle). The coverage fraction
!> (coverage / NSites) is appended as the last input, unscaled.
!==================================================================================================================================
TYPE :: t_MLP
  INTEGER                          :: NLayers
  TYPE(t_DenseLayer), ALLOCATABLE  :: Layers(:)
  REAL(wp), ALLOCATABLE            :: InputMean(:)   !> (NCond)
  REAL(wp), ALLOCATABLE            :: InputStd(:)    !> (NCond)
END TYPE t_MLP

!==================================================================================================================================
!> The "nn_ensemble" scheme: mean of the member softmax probabilities, the
!> log(coverage fraction) offset on OffsetClass, and a ceiling grid over
!> (velocity, angle) interpolated multilinearly. Mirrors nn.py's
!> NNEnsembleCoverageModel.
!==================================================================================================================================
TYPE :: t_NNModel
  INTEGER                        :: NMembers
  TYPE(t_MLP), ALLOCATABLE       :: Members(:)
  INTEGER                        :: OffsetClass       !> 1-based class index receiving the log(coverage fraction) offset
  REAL(wp)                       :: OffsetFloor       !> floor of the coverage fraction inside the log
  REAL(wp)                       :: CoverageMax
  LOGICAL                        :: HasCeiling
  INTEGER                        :: NCeilingAxes
  TYPE(t_GridAxis), ALLOCATABLE  :: CeilingAxes(:)
  INTEGER, ALLOCATABLE           :: CeilingShape(:)   !> (NCeilingAxes)
  REAL(wp), ALLOCATABLE          :: CeilingGrid(:)    !> flattened, C-order
END TYPE t_NNModel

!==================================================================================================================================
!> Top-level loaded model: the stored scheme plus the coverage physics and
!> metadata every scheme shares. Mirrors coverage_model.py's CoveragePhysics
!> plus whichever model class export_hdf5.py wrote.
!==================================================================================================================================
TYPE :: t_CoverageModel
  CHARACTER(LEN=NAME_LEN)  :: Scheme                !> "grid" or "nn_ensemble"
  CHARACTER(LEN=NAME_LEN)  :: FormatVersion
  INTEGER                  :: NClasses
  INTEGER                  :: NSites
  INTEGER                  :: ZeroedZeroIndex        !> 1-based class index impossible at coverage == 0
  INTEGER                  :: ZeroedFullIndex        !> 1-based class index impossible at coverage >= NSites
  INTEGER                  :: AbsorbingFullIndex     !> 1-based class absorbing ZeroedFullIndex's freed mass
  REAL(wp), ALLOCATABLE    :: CoverageDelta(:)       !> (NClasses)
  LOGICAL                  :: HasModelArea
  REAL(wp)                 :: ModelAreaNm2           !> area of the training surface patch (nm^2)
  TYPE(t_GridModel)        :: Grid                   !> valid iff Scheme == "grid"
  TYPE(t_NNModel)          :: NN                     !> valid iff Scheme == "nn_ensemble"
END TYPE t_CoverageModel

END MODULE MOD_CoverageKernel_Types
