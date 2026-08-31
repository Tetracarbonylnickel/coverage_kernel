!==================================================================================================================================
!> DOMAIN layer + public API: everything a DSMC boundary condition needs
!> to load a fitted coverage-dependent surface reaction model and query
!> it. This is the only module a caller (e.g. PICLas) should USE directly.
!>
!> Owns the pieces that are genuinely coverage-specific physics rather
!> than generic classifier/interpolation math (see MOD_CoverageKernel_Types'
!> header comment for the generic/domain split rationale):
!>   - the zero-at-zero-coverage / zero-at-full-coverage gates and the
!>     linear blend toward the physically-certain full-coverage endpoint
!>     (mirrors coverage_model.py's enforce_zero_at_zero_feature /
!>     enforce_zero_at_full_feature / blend_beyond_sampled_range);
!>   - this project's own (coverage, E_n, v_parallel) feature engineering
!>     for the pooled scheme (mirrors examples/features.py's
!>     physics_features -- a project-specific convenience, not something
!>     the generic core needs to know about);
!>   - inverse-CDF outcome sampling and the areal-density -> training-patch
!>     coverage stochastic-rounding conversion.
!>
!> Every dummy-argument INTENT(IN)/(OUT) here uses 1-based Fortran
!> indexing throughout (Outcome, class indices baked into the loaded
!> Model) -- the only 0-based/1-based conversion in this whole port
!> happens once, inside MOD_CoverageKernel_IO at load time.
!==================================================================================================================================
MODULE MOD_CoverageKernel
! MODULES
USE MOD_CoverageKernel_Types,      ONLY: wp, t_CoverageModel, t_GridCell
USE MOD_CoverageKernel_IO,         ONLY: ReadCoverageModelHDF5
USE MOD_CoverageKernel_Classifier, ONLY: EvalPooledClassifier
USE MOD_CoverageKernel_Grid,       ONLY: EvalGridModel
USE MOD_CoverageKernel_Interp1D,   ONLY: MultilinearClipped
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
REAL(wp), PARAMETER :: DEG2RAD = ACOS(-1.0_wp) / 180.0_wp

PUBLIC :: wp
PUBLIC :: t_CoverageModel
PUBLIC :: LoadCoverageModel
PUBLIC :: EvalCoverageProbs
PUBLIC :: SampleCoverageOutcome
PUBLIC :: MapAreaDensityToCoverage

ABSTRACT INTERFACE
  FUNCTION RawProbsInterface(FeatureValue, NClasses) RESULT(Probs)
    IMPORT :: wp
    REAL(wp), INTENT(IN) :: FeatureValue
    INTEGER,  INTENT(IN) :: NClasses
    REAL(wp) :: Probs(NClasses)
  END FUNCTION RawProbsInterface
END INTERFACE
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> Load a fitted coverage model from an HDF5 file written by
!> coverage_kernel/python/coverage_kernel/export_hdf5.py. IErr: if
!> present, receives 0 on success or nonzero if the file could not be
!> opened; if omitted and loading fails, ERROR STOP with a descriptive
!> message (see MOD_CoverageKernel_IO's ReadCoverageModelHDF5 docstring
!> for which failure modes are catchable via IErr vs. always fatal).
!==================================================================================================================================
SUBROUTINE LoadCoverageModel(FileName, Model, IErr)
! INPUT / OUTPUT VARIABLES
CHARACTER(LEN=*),      INTENT(IN)            :: FileName
TYPE(t_CoverageModel), INTENT(OUT)           :: Model
INTEGER,               INTENT(OUT), OPTIONAL :: IErr
! LOCAL VARIABLES
INTEGER :: LocalIErr
!----------------------------------------------------------------------------------------------------------------------------
CALL ReadCoverageModelHDF5(FileName, Model, LocalIErr)
IF (PRESENT(IErr)) THEN
  IErr = LocalIErr
ELSE IF (LocalIErr /= 0) THEN
  WRITE(*,*) "MOD_CoverageKernel: failed to load coverage model from ", TRIM(FileName)
  ERROR STOP "MOD_CoverageKernel: LoadCoverageModel failed (see message above)"
END IF
END SUBROUTINE LoadCoverageModel

!==================================================================================================================================
!> Evaluate P(outcome=k | Coverage, Angle, Velocity), k = 1..Model%NClasses.
!> Dispatches to the pooled or grid scheme depending on how Model was
!> fitted; the caller doesn't need to know which.
!==================================================================================================================================
SUBROUTINE EvalCoverageProbs(Model, Coverage, Angle, Velocity, Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_CoverageModel), INTENT(IN)  :: Model
REAL(wp),              INTENT(IN)  :: Coverage, Angle, Velocity
REAL(wp),              INTENT(OUT) :: Probs(Model%NClasses)
! LOCAL VARIABLES
REAL(wp) :: Ceiling, AxisQuery(2)
LOGICAL  :: HasCeiling
INTEGER  :: EmptyIdx(0)
!----------------------------------------------------------------------------------------------------------------------------
SELECT CASE (TRIM(Model%Scheme))
CASE ("pooled")
  HasCeiling = Model%Pooled%HasCeiling
  Ceiling = 0.0_wp
  IF (HasCeiling) THEN
    Ceiling = MultilinearClipped(Model%Pooled%CeilingAxes, Model%Pooled%CeilingShape, &
                                  Model%Pooled%CeilingGrid, [Velocity, Angle], 1, EmptyIdx)
  END IF
  Probs = BlendBeyondSampledRange(PooledRawProbs, Coverage, Model%Pooled%CoverageMax, REAL(Model%NSites, wp), &
                                   Model%ZeroedFullIndex, Model%AbsorbingFullIndex, HasCeiling, Ceiling, Model%NClasses)

CASE ("grid")
  AxisQuery = [Velocity, Angle]
  Probs = EvalGridModel(Model%Grid%Axes, Model%Grid%Shape, Model%Grid%InterpMethod, Coverage, AxisQuery, &
                         CellEvalLeaf, Model%NClasses)

CASE DEFAULT
  ERROR STOP "MOD_CoverageKernel: unknown model scheme (expected 'pooled' or 'grid')"
END SELECT

CONTAINS

  !> Pooled scheme's raw (pre-blend) probability at a given coverage value:
  !> engineer (coverage, E_n, v_parallel) from (FeatureValue, Velocity,
  !> Angle) via this project's own feature decomposition, evaluate the
  !> shared classifier, then apply the zero-at-zero-coverage gate.
  !> Host-associated to Model/Angle/Velocity (both fixed for this call to
  !> EvalCoverageProbs); passed to BlendBeyondSampledRange as its raw-eval
  !> callback.
  FUNCTION PooledRawProbs(FeatureValue, NClasses) RESULT(P)
  REAL(wp), INTENT(IN) :: FeatureValue
  INTEGER,  INTENT(IN) :: NClasses
  REAL(wp) :: P(NClasses)
  REAL(wp) :: AngleRad, VNormal, VParallel, XRaw(3)
  AngleRad  = Angle * DEG2RAD
  VNormal   = Velocity * COS(AngleRad)
  VParallel = Velocity * SIN(AngleRad)
  XRaw = [FeatureValue, VNormal**2, VParallel]
  P = EvalPooledClassifier(Model%Pooled%Classifier, XRaw)
  P = EnforceZeroAtZeroFeature(P, FeatureValue, Model%ZeroedZeroIndex)
  END FUNCTION PooledRawProbs

  !> Grid scheme's per-cell leaf evaluator (matches Grid's CellEvalInterface):
  !> look up the cell at FlatIdx and evaluate its own already-fitted
  !> classifier blended with its own coverage_max/ceiling, exactly the
  !> order coverage_model.py's CoverageCell enforces (blend per cell,
  !> before MOD_CoverageKernel_Grid interpolates across cells). Delegates
  !> to the plain (non-closure) BlendGridCell/GridCellRaw module
  !> functions below rather than BlendBeyondSampledRange's callback
  !> machinery: Fortran does not allow an internal procedure (this one) to
  !> contain a further internal procedure of its own, so the per-cell raw
  !> evaluator cannot itself be a host-associated closure here the way
  !> PooledRawProbs is one level up.
  FUNCTION CellEvalLeaf(FlatIdx, FeatureValue, NClasses) RESULT(P)
  INTEGER,  INTENT(IN) :: FlatIdx, NClasses
  REAL(wp), INTENT(IN) :: FeatureValue
  REAL(wp) :: P(NClasses)
  P = BlendGridCell(Model%Grid%Cells(FlatIdx), FeatureValue, Model%NSites, &
                     Model%ZeroedZeroIndex, Model%ZeroedFullIndex, Model%AbsorbingFullIndex, NClasses)
  END FUNCTION CellEvalLeaf

END SUBROUTINE EvalCoverageProbs

!==================================================================================================================================
!> Sample an outcome from EvalCoverageProbs's own probability vector via
!> inverse-CDF, using caller-supplied Xi in [0,1) (this module never owns
!> an RNG -- see MOD_CoverageKernel_Types' header and the project's public
!> API design). Returns both Outcome and the Probs it was drawn from, and
!> DeltaCoverage = Model%CoverageDelta(Outcome) as a convenience.
!==================================================================================================================================
SUBROUTINE SampleCoverageOutcome(Model, Coverage, Angle, Velocity, Xi, Outcome, DeltaCoverage, Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_CoverageModel), INTENT(IN)  :: Model
REAL(wp),              INTENT(IN)  :: Coverage, Angle, Velocity, Xi
INTEGER,               INTENT(OUT) :: Outcome
REAL(wp),              INTENT(OUT) :: DeltaCoverage
REAL(wp),              INTENT(OUT) :: Probs(Model%NClasses)
!----------------------------------------------------------------------------------------------------------------------------
CALL EvalCoverageProbs(Model, Coverage, Angle, Velocity, Probs)
Outcome = SampleOutcome(Probs, Xi)
DeltaCoverage = Model%CoverageDelta(Outcome)
END SUBROUTINE SampleCoverageOutcome

!==================================================================================================================================
!> Map an areal adsorbate density on a surface element of area ElementArea
!> to an integer coverage on the model's own training-patch area
!> (Model%ModelAreaNm2), via stochastic rounding of the expected count
!> (mirrors coverage_model.py's map_area_density_to_coverage /
!> stochastic_round). Requires the loaded model to carry a
!> model_area_nm2 (i.e. it was exported with one); this is a hard
!> precondition, not a recoverable runtime case, since it means the
!> model file simply wasn't built for density-based queries.
!==================================================================================================================================
FUNCTION MapAreaDensityToCoverage(Model, AreaDensity, ElementArea, Xi) RESULT(LocalCoverage)
! INPUT / OUTPUT VARIABLES
TYPE(t_CoverageModel), INTENT(IN) :: Model
REAL(wp),              INTENT(IN) :: AreaDensity, ElementArea, Xi
INTEGER :: LocalCoverage
! LOCAL VARIABLES
REAL(wp) :: LocalDensity, LocalCoverageReal, Frac
INTEGER  :: FloorVal
!----------------------------------------------------------------------------------------------------------------------------
IF (.NOT. Model%HasModelArea) THEN
  ERROR STOP "MOD_CoverageKernel: MapAreaDensityToCoverage requires a model exported with model_area_nm2"
END IF
LocalDensity = AreaDensity / ElementArea
LocalCoverageReal = LocalDensity * Model%ModelAreaNm2
FloorVal = FLOOR(LocalCoverageReal)
Frac = LocalCoverageReal - REAL(FloorVal, wp)
LocalCoverage = FloorVal
IF (Xi < Frac) LocalCoverage = LocalCoverage + 1
END FUNCTION MapAreaDensityToCoverage

!==================================================================================================================================
!> Inverse-CDF categorical sample: smallest 1-based index k with
!> Xi <= cumsum(Probs)(k), matching NumPy's
!> searchsorted(cumsum(probs), xi, side="right") exactly (count of
!> cumulative sums <= Xi, then +1 for 1-based indexing, clamped to
!> NClasses so floating-point roundoff at Xi~1.0 can't run past the end).
!==================================================================================================================================
FUNCTION SampleOutcome(Probs, Xi) RESULT(Idx)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: Probs(:)
REAL(wp), INTENT(IN) :: Xi
INTEGER :: Idx
! LOCAL VARIABLES
REAL(wp) :: CumArr(SIZE(Probs))
INTEGER  :: k
!----------------------------------------------------------------------------------------------------------------------------
CumArr(1) = Probs(1)
DO k = 2, SIZE(Probs)
  CumArr(k) = CumArr(k - 1) + Probs(k)
END DO
Idx = MIN(COUNT(CumArr <= Xi) + 1, SIZE(Probs))
END FUNCTION SampleOutcome

!==================================================================================================================================
!> Hard-zero Probs(ZeroedIndex) when FeatureValue == 0, renormalizing the
!> rest. Mirrors coverage_model.py's enforce_zero_at_zero_feature.
!==================================================================================================================================
FUNCTION EnforceZeroAtZeroFeature(Probs, FeatureValue, ZeroedIndex) RESULT(Out)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: Probs(:)
REAL(wp), INTENT(IN) :: FeatureValue
INTEGER,  INTENT(IN) :: ZeroedIndex
REAL(wp) :: Out(SIZE(Probs))
! LOCAL VARIABLES
REAL(wp) :: Total
!----------------------------------------------------------------------------------------------------------------------------
Out = Probs
IF (FeatureValue == 0.0_wp .AND. Out(ZeroedIndex) /= 0.0_wp) THEN
  Out(ZeroedIndex) = 0.0_wp
  Total = SUM(Out)
  IF (Total > 0.0_wp) Out = Out / Total
END IF
END FUNCTION EnforceZeroAtZeroFeature

!==================================================================================================================================
!> Hard-zero Probs(ZeroedIndex) once FeatureValue >= FeatureCap, handing
!> all of its mass to Probs(AbsorbingIndex) (every other class untouched).
!> Mirrors coverage_model.py's enforce_zero_at_full_feature.
!==================================================================================================================================
FUNCTION EnforceZeroAtFullFeature(Probs, FeatureValue, FeatureCap, ZeroedIndex, AbsorbingIndex) RESULT(Out)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: Probs(:)
REAL(wp), INTENT(IN) :: FeatureValue, FeatureCap
INTEGER,  INTENT(IN) :: ZeroedIndex, AbsorbingIndex
REAL(wp) :: Out(SIZE(Probs))
!----------------------------------------------------------------------------------------------------------------------------
Out = Probs
IF (FeatureValue >= FeatureCap .AND. Out(ZeroedIndex) /= 0.0_wp) THEN
  Out(AbsorbingIndex) = Out(AbsorbingIndex) + Out(ZeroedIndex)
  Out(ZeroedIndex) = 0.0_wp
END IF
END FUNCTION EnforceZeroAtFullFeature

!==================================================================================================================================
!> Trust RawProbsFn as-is up to BlendLo = MIN(TrustedMax, Ceiling-or-Cap),
!> then linearly blend toward the fully-saturated endpoint (built by
!> EnforceZeroAtFullFeature at BlendLo) out to
!> BlendHi = MAX(TrustedMax, Ceiling-or-Cap). HasCeiling=.FALSE.
!> reproduces blending all the way to FeatureCap (the "everyday" case
!> where the condition's own dynamics settle beyond the dataset-wide
!> trusted range). Mirrors coverage_model.py's blend_beyond_sampled_range.
!> Used by the pooled scheme only -- see EvalCoverageProbs/CellEvalLeaf's
!> docstring for why the grid scheme's per-cell blend uses the plain
!> (non-callback) BlendGridCell instead.
!==================================================================================================================================
FUNCTION BlendBeyondSampledRange(RawProbsFn, FeatureValue, TrustedMax, FeatureCap, ZeroedIndex, AbsorbingIndex, &
                                  HasCeiling, Ceiling, NClasses) RESULT(Probs)
! INPUT / OUTPUT VARIABLES
PROCEDURE(RawProbsInterface)  :: RawProbsFn
REAL(wp), INTENT(IN) :: FeatureValue, TrustedMax, FeatureCap
INTEGER,  INTENT(IN) :: ZeroedIndex, AbsorbingIndex, NClasses
LOGICAL,  INTENT(IN) :: HasCeiling
REAL(wp), INTENT(IN) :: Ceiling
REAL(wp) :: Probs(NClasses)
! LOCAL VARIABLES
REAL(wp) :: C, BlendLo, BlendHi, T
REAL(wp) :: PBoundary(NClasses), PEnd(NClasses)
!----------------------------------------------------------------------------------------------------------------------------
C = MERGE(Ceiling, FeatureCap, HasCeiling)
BlendLo = MIN(TrustedMax, C)
BlendHi = MAX(TrustedMax, C)

IF (FeatureValue <= BlendLo) THEN
  Probs = RawProbsFn(FeatureValue, NClasses)
  RETURN
END IF

PBoundary = RawProbsFn(BlendLo, NClasses)
PEnd = EnforceZeroAtFullFeature(PBoundary, FeatureCap, FeatureCap, ZeroedIndex, AbsorbingIndex)

IF (FeatureValue >= BlendHi) THEN
  Probs = PEnd
  RETURN
END IF

T = (FeatureValue - BlendLo) / (BlendHi - BlendLo)
Probs = (1.0_wp - T) * PBoundary + T * PEnd
END FUNCTION BlendBeyondSampledRange

!==================================================================================================================================
!> One grid cell's raw (pre-blend) probability at a given coverage value:
!> the cell's own classifier takes coverage as its sole raw feature,
!> followed by the zero-at-zero-coverage gate. Plain module function
!> (no host association needed) so it can be called directly from
!> BlendGridCell without requiring a nested internal-procedure closure.
!==================================================================================================================================
FUNCTION GridCellRaw(Cell, FeatureValue, ZeroedZeroIndex, NClasses) RESULT(Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_GridCell), INTENT(IN) :: Cell
REAL(wp),         INTENT(IN) :: FeatureValue
INTEGER,          INTENT(IN) :: ZeroedZeroIndex, NClasses
REAL(wp) :: Probs(NClasses)
! LOCAL VARIABLES
REAL(wp) :: XRaw(1)
!----------------------------------------------------------------------------------------------------------------------------
XRaw(1) = FeatureValue
Probs = EvalPooledClassifier(Cell%Classifier, XRaw)
Probs = EnforceZeroAtZeroFeature(Probs, FeatureValue, ZeroedZeroIndex)
END FUNCTION GridCellRaw

!==================================================================================================================================
!> Same blend policy as BlendBeyondSampledRange, specialized (in plain
!> argument-passing form, no callback) to one grid cell's own
!> CoverageMax/Ceiling -- see CellEvalLeaf's docstring for why this isn't
!> just a call to BlendBeyondSampledRange.
!==================================================================================================================================
FUNCTION BlendGridCell(Cell, FeatureValue, NSites, ZeroedZeroIndex, ZeroedFullIndex, AbsorbingFullIndex, NClasses) &
    RESULT(Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_GridCell), INTENT(IN) :: Cell
REAL(wp),         INTENT(IN) :: FeatureValue
INTEGER,          INTENT(IN) :: NSites, ZeroedZeroIndex, ZeroedFullIndex, AbsorbingFullIndex, NClasses
REAL(wp) :: Probs(NClasses)
! LOCAL VARIABLES
REAL(wp) :: Cap, C, BlendLo, BlendHi, T
REAL(wp) :: PBoundary(NClasses), PEnd(NClasses)
!----------------------------------------------------------------------------------------------------------------------------
Cap = REAL(NSites, wp)
C = MERGE(Cell%Ceiling, Cap, Cell%HasCeiling)
BlendLo = MIN(Cell%CoverageMax, C)
BlendHi = MAX(Cell%CoverageMax, C)

IF (FeatureValue <= BlendLo) THEN
  Probs = GridCellRaw(Cell, FeatureValue, ZeroedZeroIndex, NClasses)
  RETURN
END IF

PBoundary = GridCellRaw(Cell, BlendLo, ZeroedZeroIndex, NClasses)
PEnd = EnforceZeroAtFullFeature(PBoundary, Cap, Cap, ZeroedFullIndex, AbsorbingFullIndex)

IF (FeatureValue >= BlendHi) THEN
  Probs = PEnd
  RETURN
END IF

T = (FeatureValue - BlendLo) / (BlendHi - BlendLo)
Probs = (1.0_wp - T) * PBoundary + T * PEnd
END FUNCTION BlendGridCell

END MODULE MOD_CoverageKernel
