!==================================================================================================================================
!> DOMAIN layer + public API: everything a DSMC boundary condition needs to
!> load a fitted coverage-dependent branching model and query it. This is
!> the only module a caller (e.g. PICLas) should USE directly.
!>
!> Two model schemes, chosen by the file:
!>   - "nn_ensemble": mean of feed-forward networks of (velocity, angle,
!>     coverage / NSites) with a log(coverage fraction) offset on one class
!>     (mirrors nn.py's NNEnsembleCoverageModel);
!>   - "grid": one spline classifier of coverage per (velocity, angle) cell,
!>     interpolated across cells (mirrors grid_classifier.py's GridClassifier).
!>
!> Coverage physics shared by both (mirrors coverage_model.py):
!>   - zero-at-zero-coverage gate: ZeroedZeroIndex has probability 0 at coverage 0;
!>   - blend beyond the sampled coverage range: the fitted model is used up to
!>     BlendLo and blended linearly toward the no-adsorption endpoint (the
!>     ZeroedFullIndex mass moved to AbsorbingFullIndex) up to BlendHi, see
!>     BlendLimits; the ceiling of the NN is interpolated multilinearly
!>     over (velocity, angle), the grid has one per cell;
!>   - inverse-CDF outcome sampling and the adsorbate-count -> model-patch
!>     coverage conversion by stochastic rounding.
!>
!> Argument order everywhere: (Coverage, Velocity, Angle), as in Python.
!> Velocity in m/s, Angle in degrees from the surface normal, Coverage in
!> adsorbed atoms on the model's training patch.
!>
!> Every index in the API is 1-based; the 0-based -> 1-based conversion of
!> the file happens once, in MOD_CoverageKernel_IO. No procedure here passes
!> an internal procedure as an argument, so no executable-stack trampolines
!> are generated.
!==================================================================================================================================
MODULE MOD_CoverageKernel
! MODULES
USE MOD_CoverageKernel_Types,      ONLY: wp, t_CoverageModel, t_GridCell
USE MOD_CoverageKernel_IO,         ONLY: ReadCoverageModelHDF5
USE MOD_CoverageKernel_Classifier, ONLY: EvalSplineClassifier
USE MOD_CoverageKernel_Grid,       ONLY: EvalGridModel
USE MOD_CoverageKernel_Interp1D,   ONLY: MultilinearClipped
USE MOD_CoverageKernel_MLP,        ONLY: EvalMLP
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
PUBLIC :: wp
PUBLIC :: t_CoverageModel
PUBLIC :: LoadCoverageModel
PUBLIC :: EvalCoverageProbs
PUBLIC :: SampleCoverageOutcome
PUBLIC :: MapAdsorbateCountToCoverage
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> Load a fitted model from an HDF5 file written by export_hdf5.py.
!> IErr: if present, 0 on success, 1 if the file does not exist, 2 if HDF5
!> cannot open it; if omitted, any of these stops with a message. A file
!> that opens but does not match the schema always stops (deployment error).
!==================================================================================================================================
SUBROUTINE LoadCoverageModel(FileName, Model, IErr)
! INPUT / OUTPUT VARIABLES
CHARACTER(LEN=*),      INTENT(IN)            :: FileName
TYPE(t_CoverageModel), INTENT(OUT)           :: Model
INTEGER,               INTENT(OUT), OPTIONAL :: IErr
! LOCAL VARIABLES
INTEGER :: LocalIErr, m
!----------------------------------------------------------------------------------------------------------------------------
CALL ReadCoverageModelHDF5(FileName, Model, LocalIErr)
IF (PRESENT(IErr)) IErr = LocalIErr
IF (LocalIErr /= 0) THEN
  IF (PRESENT(IErr)) RETURN
  WRITE(*,*) "MOD_CoverageKernel: cannot open coverage model file ", TRIM(FileName)
  ERROR STOP "MOD_CoverageKernel: LoadCoverageModel failed (see message above)"
END IF

! the domain layer evaluates models of (coverage, velocity, angle)
SELECT CASE (TRIM(Model%Scheme))
CASE ("grid")
  IF (Model%Grid%NAxes /= 2) ERROR STOP "MOD_CoverageKernel: grid model must have 2 axes (velocity, angle)"
CASE ("nn_ensemble")
  DO m = 1, Model%NN%NMembers
    IF (SIZE(Model%NN%Members(m)%InputMean) /= 2) &
      ERROR STOP "MOD_CoverageKernel: networks must have 2 conditioning inputs (velocity, angle)"
  END DO
  IF (Model%NN%HasCeiling .AND. Model%NN%NCeilingAxes /= 2) &
    ERROR STOP "MOD_CoverageKernel: NN ceiling grid must have 2 axes (velocity, angle)"
END SELECT
END SUBROUTINE LoadCoverageModel

!==================================================================================================================================
!> P(outcome = k | Coverage, Velocity, Angle), k = 1..Model%NClasses.
!==================================================================================================================================
SUBROUTINE EvalCoverageProbs(Model, Coverage, Velocity, Angle, Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_CoverageModel), INTENT(IN)  :: Model
REAL(wp),              INTENT(IN)  :: Coverage, Velocity, Angle
REAL(wp),              INTENT(OUT) :: Probs(Model%NClasses)
! LOCAL VARIABLES
REAL(wp) :: Ceiling, BlendLo, BlendHi
INTEGER  :: EmptyIdx(0)
!----------------------------------------------------------------------------------------------------------------------------
SELECT CASE (TRIM(Model%Scheme))
CASE ("nn_ensemble")
  Ceiling = 0.0_wp
  IF (Model%NN%HasCeiling) Ceiling = MultilinearClipped(Model%NN%CeilingAxes, Model%NN%CeilingShape, &
                                                        Model%NN%CeilingGrid, [Velocity, Angle], 1, EmptyIdx)
  CALL BlendLimits(Model%NN%CoverageMax, Model%NN%HasCeiling, Ceiling, REAL(Model%NSites, wp), BlendLo, BlendHi)
  Probs = NNRawProbs(Model, MIN(Coverage, BlendLo), Velocity, Angle)
  Probs = BlendFromBoundary(Probs, Coverage, BlendLo, BlendHi, REAL(Model%NSites, wp), &
                            Model%ZeroedFullIndex, Model%AbsorbingFullIndex)
CASE ("grid")
  Probs = EvalGridModel(Model, Model%Grid%Axes, Model%Grid%Shape, Model%Grid%InterpMethod, Coverage, &
                        [Velocity, Angle], GridCellLeaf, Model%NClasses)
CASE DEFAULT
  ERROR STOP "MOD_CoverageKernel: unknown model scheme (expected 'grid' or 'nn_ensemble')"
END SELECT
END SUBROUTINE EvalCoverageProbs

!==================================================================================================================================
!> Sample an outcome by inverse CDF with the caller's uniform deviate Xi in
!> [0,1) (this module never owns a random-number generator; the host keeps
!> its own stream). Returns the 1-based Outcome, the probabilities it was
!> drawn from, and DeltaCoverage = Model%CoverageDelta(Outcome).
!==================================================================================================================================
SUBROUTINE SampleCoverageOutcome(Model, Coverage, Velocity, Angle, Xi, Outcome, DeltaCoverage, Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_CoverageModel), INTENT(IN)  :: Model
REAL(wp),              INTENT(IN)  :: Coverage, Velocity, Angle, Xi
INTEGER,               INTENT(OUT) :: Outcome
REAL(wp),              INTENT(OUT) :: DeltaCoverage
REAL(wp),              INTENT(OUT) :: Probs(Model%NClasses)
!----------------------------------------------------------------------------------------------------------------------------
CALL EvalCoverageProbs(Model, Coverage, Velocity, Angle, Probs)
Outcome = SampleOutcome(Probs, Xi)
DeltaCoverage = Model%CoverageDelta(Outcome)
END SUBROUTINE SampleCoverageOutcome

!==================================================================================================================================
!> Map the number of adsorbed atoms AdsorbateCount on a surface element of
!> area ElementArea (nm^2) to an integer coverage on the model's training
!> patch (Model%ModelAreaNm2), by stochastic rounding of
!> AdsorbateCount / ElementArea * ModelAreaNm2 with the caller's deviate Xi.
!> Mirrors coverage_model.py's map_adsorbate_count_to_coverage. Requires a
!> model exported with model_area_nm2.
!==================================================================================================================================
FUNCTION MapAdsorbateCountToCoverage(Model, AdsorbateCount, ElementArea, Xi) RESULT(LocalCoverage)
! INPUT / OUTPUT VARIABLES
TYPE(t_CoverageModel), INTENT(IN) :: Model
REAL(wp),              INTENT(IN) :: AdsorbateCount, ElementArea, Xi
INTEGER :: LocalCoverage
! LOCAL VARIABLES
REAL(wp) :: LocalCoverageReal, Frac
INTEGER  :: FloorVal
!----------------------------------------------------------------------------------------------------------------------------
IF (.NOT. Model%HasModelArea) &
  ERROR STOP "MOD_CoverageKernel: MapAdsorbateCountToCoverage requires a model exported with model_area_nm2"
LocalCoverageReal = AdsorbateCount / ElementArea * Model%ModelAreaNm2
FloorVal = FLOOR(LocalCoverageReal)
Frac = LocalCoverageReal - REAL(FloorVal, wp)
LocalCoverage = FloorVal
IF (Xi < Frac) LocalCoverage = LocalCoverage + 1
END FUNCTION MapAdsorbateCountToCoverage

!==================================================================================================================================
!> Inverse-CDF categorical sample: smallest 1-based index k with
!> Xi < cumsum(Probs)(k), i.e. NumPy's searchsorted(cumsum, xi, side="right")
!> + 1, clamped to SIZE(Probs) against roundoff at Xi close to 1.
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
!> Coverage range of the blend toward the no-adsorption endpoint. Mirrors
!> coverage_model.py's blend_limits. TrustedMax is the highest sampled
!> coverage, Ceiling the self-consistent ceiling (if any), Cap = NSites.
!> The fitted model is always used up to TrustedMax; a ceiling at or below
!> TrustedMax makes the endpoint apply immediately above it.
!==================================================================================================================================
SUBROUTINE BlendLimits(TrustedMax, HasCeiling, Ceiling, Cap, BlendLo, BlendHi)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN)  :: TrustedMax, Ceiling, Cap
LOGICAL,  INTENT(IN)  :: HasCeiling
REAL(wp), INTENT(OUT) :: BlendLo, BlendHi
! LOCAL VARIABLES
REAL(wp) :: C
!----------------------------------------------------------------------------------------------------------------------------
C = MERGE(Ceiling, Cap, HasCeiling)
BlendLo = TrustedMax
BlendHi = MAX(TrustedMax, C)
END SUBROUTINE BlendLimits

!==================================================================================================================================
!> Blend given the raw probabilities PAtQuery evaluated at
!> MIN(Coverage, BlendLo): unchanged up to BlendLo; beyond it a linear
!> blend from PAtQuery (= the raw probabilities at BlendLo) toward the
!> endpoint built by EnforceZeroAtFullFeature, reached at BlendHi.
!> Mirrors coverage_model.py's blend_beyond_sampled_range.
!==================================================================================================================================
FUNCTION BlendFromBoundary(PAtQuery, Coverage, BlendLo, BlendHi, Cap, ZeroedIndex, AbsorbingIndex) RESULT(Probs)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: PAtQuery(:)
REAL(wp), INTENT(IN) :: Coverage, BlendLo, BlendHi, Cap
INTEGER,  INTENT(IN) :: ZeroedIndex, AbsorbingIndex
REAL(wp) :: Probs(SIZE(PAtQuery))
! LOCAL VARIABLES
REAL(wp) :: PEnd(SIZE(PAtQuery)), T
!----------------------------------------------------------------------------------------------------------------------------
IF (Coverage <= BlendLo) THEN
  Probs = PAtQuery
  RETURN
END IF
PEnd = EnforceZeroAtFullFeature(PAtQuery, Cap, Cap, ZeroedIndex, AbsorbingIndex)
IF (Coverage >= BlendHi) THEN
  Probs = PEnd
  RETURN
END IF
T = (Coverage - BlendLo) / (BlendHi - BlendLo)
Probs = (1.0_wp - T) * PAtQuery + T * PEnd
END FUNCTION BlendFromBoundary

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
!> Hard-zero Probs(ZeroedIndex) once FeatureValue >= FeatureCap, moving its
!> mass to Probs(AbsorbingIndex). Mirrors coverage_model.py's
!> enforce_zero_at_full_feature.
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
!> Raw (pre-blend) ensemble probabilities: for every member, the network
!> output with log(MAX(Coverage / NSites, OffsetFloor)) added to the offset
!> class, softmax, zero-at-zero-coverage gate; then the mean over members.
!> Mirrors nn.py's NNEnsemble.probs_batch.
!==================================================================================================================================
FUNCTION NNRawProbs(Model, Coverage, Velocity, Angle) RESULT(Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_CoverageModel), INTENT(IN) :: Model
REAL(wp),              INTENT(IN) :: Coverage, Velocity, Angle
REAL(wp) :: Probs(Model%NClasses)
! LOCAL VARIABLES
REAL(wp) :: X(3), Z(Model%NClasses), P(Model%NClasses), Fraction
INTEGER  :: m
!----------------------------------------------------------------------------------------------------------------------------
Fraction = Coverage / REAL(Model%NSites, wp)
Probs = 0.0_wp
DO m = 1, Model%NN%NMembers
  ASSOCIATE(Net => Model%NN%Members(m))
    X(1:2) = ([Velocity, Angle] - Net%InputMean) / Net%InputStd
    X(3) = Fraction
    Z = EvalMLP(Net, X)
  END ASSOCIATE
  Z(Model%NN%OffsetClass) = Z(Model%NN%OffsetClass) + LOG(MAX(Fraction, Model%NN%OffsetFloor))
  P = EXP(Z - MAXVAL(Z))
  P = P / SUM(P)
  IF (Coverage == 0.0_wp) THEN
    P(Model%ZeroedZeroIndex) = 0.0_wp
    P = P / SUM(P)
  END IF
  Probs = Probs + P
END DO
Probs = Probs / REAL(Model%NN%NMembers, wp)
END FUNCTION NNRawProbs

!==================================================================================================================================
!> Grid scheme leaf evaluator (MOD_CoverageKernel_Grid's CellEvalInterface):
!> the cell's own classifier with the zero-at-zero gate, blended with the
!> cell's own CoverageMax/Ceiling -- blend per cell, then interpolation
!> across cells, as coverage_model.py's CoverageCell.
!==================================================================================================================================
FUNCTION GridCellLeaf(Model, FlatIdx, Feature, NClasses) RESULT(Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_CoverageModel), INTENT(IN) :: Model
INTEGER,               INTENT(IN) :: FlatIdx, NClasses
REAL(wp),              INTENT(IN) :: Feature
REAL(wp) :: Probs(NClasses)
! LOCAL VARIABLES
REAL(wp) :: BlendLo, BlendHi
!----------------------------------------------------------------------------------------------------------------------------
ASSOCIATE(Cell => Model%Grid%Cells(FlatIdx))
  CALL BlendLimits(Cell%CoverageMax, Cell%HasCeiling, Cell%Ceiling, REAL(Model%NSites, wp), BlendLo, BlendHi)
  Probs = GridCellRaw(Cell, MIN(Feature, BlendLo), Model%ZeroedZeroIndex, NClasses)
  Probs = BlendFromBoundary(Probs, Feature, BlendLo, BlendHi, REAL(Model%NSites, wp), &
                            Model%ZeroedFullIndex, Model%AbsorbingFullIndex)
END ASSOCIATE
END FUNCTION GridCellLeaf

!==================================================================================================================================
!> One grid cell's raw probabilities at a coverage value: the cell's
!> classifier of coverage, then the zero-at-zero-coverage gate.
!==================================================================================================================================
FUNCTION GridCellRaw(Cell, FeatureValue, ZeroedZeroIndex, NClasses) RESULT(Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_GridCell), INTENT(IN) :: Cell
REAL(wp),         INTENT(IN) :: FeatureValue
INTEGER,          INTENT(IN) :: ZeroedZeroIndex, NClasses
REAL(wp) :: Probs(NClasses)
!----------------------------------------------------------------------------------------------------------------------------
Probs = EvalSplineClassifier(Cell%Classifier, [FeatureValue])
Probs = EnforceZeroAtZeroFeature(Probs, FeatureValue, ZeroedZeroIndex)
END FUNCTION GridCellRaw

END MODULE MOD_CoverageKernel
