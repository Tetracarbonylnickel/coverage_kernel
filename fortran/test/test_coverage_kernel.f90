!==================================================================================================================================
!> Cross-validation harness: loads the two synthetic HDF5 models built by
!> generate_reference_data.py and checks that MOD_CoverageKernel's public
!> API reproduces the Python reference implementation's numbers (within a
!> small floating-point tolerance) across a battery of query points,
!> including out-of-range/extrapolation queries, coverage==0, and the
!> deep-blend region beyond the sampled coverage range. Also checks
!> SampleCoverageOutcome's inverse-CDF sampling and
!> MapAreaDensityToCoverage's stochastic rounding.
!>
!> Not a unit test of individual generic-core pieces (those are validated
!> in isolation on the Python side already) -- this is specifically an
!> end-to-end Fortran-vs-Python parity check of the whole loaded-model
!> evaluation path.
!==================================================================================================================================
PROGRAM test_coverage_kernel
! MODULES
USE MOD_CoverageKernel
USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: REAL64
IMPLICIT NONE
!----------------------------------------------------------------------------------------------------------------------------
INTEGER,      PARAMETER :: NClassesTest = 3
REAL(REAL64), PARAMETER :: Tol = 1.0E-9_REAL64

TYPE(t_CoverageModel) :: PooledModel, GridModel, PooledRealModel, GridRealModel
INTEGER :: NFail
LOGICAL :: HaveRealModels
!==================================================================================================================================
NFail = 0

CALL LoadCoverageModel("data/pooled_test.h5", PooledModel)
CALL LoadCoverageModel("data/grid_test.h5", GridModel)

CALL CheckProbsFile("data/reference_pooled.txt", PooledModel, NFail)
CALL CheckProbsFile("data/reference_grid.txt", GridModel, NFail)
CALL CheckSampling("data/reference_sampling.txt", PooledModel, NFail)
CALL CheckDensity("data/reference_density.txt", PooledModel, NFail)

! Also cross-validate against the real production models fitted in
! train_models.ipynb on the full ~400k-event MLIP dataset (8 velocities x
! 5 angles, n_sites=48), when generate_reference_data.py found them --
! the real grid shape/ceiling grid/fitted coefficients, not just the small
! synthetic models above.
INQUIRE(FILE="data/pooled_real.h5", EXIST=HaveRealModels)
IF (HaveRealModels) THEN
  CALL LoadCoverageModel("data/pooled_real.h5", PooledRealModel)
  CALL LoadCoverageModel("data/grid_real.h5", GridRealModel)
  CALL CheckProbsFile("data/reference_pooled_real.txt", PooledRealModel, NFail)
  CALL CheckProbsFile("data/reference_grid_real.txt", GridRealModel, NFail)
ELSE
  WRITE(*, *) "Skipping real-model cross-validation: data/pooled_real.h5 not found"
END IF

IF (NFail > 0) THEN
  WRITE(*, '(A,I0,A)') "FAILED: ", NFail, " mismatch(es)"
  ERROR STOP "coverage_kernel Fortran-vs-Python cross-validation FAILED"
ELSE
  WRITE(*, *) "All coverage_kernel Fortran-vs-Python cross-validation checks PASSED"
END IF

CONTAINS

!==================================================================================================================================
!> Check every "coverage velocity angle p1 p2 p3" line of a reference
!> file against EvalCoverageProbs(Model, coverage, angle, velocity, ...).
!==================================================================================================================================
SUBROUTINE CheckProbsFile(FileName, Model, NFail)
CHARACTER(LEN=*),      INTENT(IN)    :: FileName
TYPE(t_CoverageModel), INTENT(IN)    :: Model
INTEGER,               INTENT(INOUT) :: NFail
INTEGER :: Unit, IOStat, LineNo
REAL(REAL64) :: Coverage, Velocity, Angle, MaxErr
REAL(REAL64) :: Expected(NClassesTest), Probs(NClassesTest)
!----------------------------------------------------------------------------------------------------------------------------
OPEN(NEWUNIT=Unit, FILE=FileName, STATUS="OLD", ACTION="READ")
LineNo = 0
DO
  READ(Unit, *, IOSTAT=IOStat) Coverage, Velocity, Angle, Expected
  IF (IOStat /= 0) EXIT
  LineNo = LineNo + 1
  CALL EvalCoverageProbs(Model, Coverage, Angle, Velocity, Probs)
  MaxErr = MAXVAL(ABS(Probs - Expected))
  IF (MaxErr > Tol) THEN
    NFail = NFail + 1
    WRITE(*, '(3A,I0,A,ES12.4)') "MISMATCH in ", TRIM(FileName), " line ", LineNo, ": max abs err = ", MaxErr
    WRITE(*, *) "  coverage=", Coverage, " velocity=", Velocity, " angle=", Angle
    WRITE(*, *) "  expected:", Expected
    WRITE(*, *) "  got:     ", Probs
  END IF
END DO
CLOSE(Unit)
WRITE(*, '(3A,I0,A)') "Checked ", TRIM(FileName), ": ", LineNo, " cases"
END SUBROUTINE CheckProbsFile

!==================================================================================================================================
!> Check every "coverage velocity angle xi expected_idx0 expected_delta"
!> line against SampleCoverageOutcome, converting its 1-based Outcome back
!> to 0-based for comparison against the Python-side (0-based) reference.
!==================================================================================================================================
SUBROUTINE CheckSampling(FileName, Model, NFail)
CHARACTER(LEN=*),      INTENT(IN)    :: FileName
TYPE(t_CoverageModel), INTENT(IN)    :: Model
INTEGER,               INTENT(INOUT) :: NFail
INTEGER :: Unit, IOStat, LineNo, ExpectedIdx0, Outcome
REAL(REAL64) :: Coverage, Velocity, Angle, Xi, ExpectedDelta, DeltaCoverage
REAL(REAL64) :: Probs(NClassesTest)
!----------------------------------------------------------------------------------------------------------------------------
OPEN(NEWUNIT=Unit, FILE=FileName, STATUS="OLD", ACTION="READ")
LineNo = 0
DO
  READ(Unit, *, IOSTAT=IOStat) Coverage, Velocity, Angle, Xi, ExpectedIdx0, ExpectedDelta
  IF (IOStat /= 0) EXIT
  LineNo = LineNo + 1
  CALL SampleCoverageOutcome(Model, Coverage, Angle, Velocity, Xi, Outcome, DeltaCoverage, Probs)
  IF ((Outcome - 1) /= ExpectedIdx0 .OR. ABS(DeltaCoverage - ExpectedDelta) > Tol) THEN
    NFail = NFail + 1
    WRITE(*, '(3A,I0)') "MISMATCH in ", TRIM(FileName), " line ", LineNo
    WRITE(*, *) "  expected idx0=", ExpectedIdx0, " delta=", ExpectedDelta
    WRITE(*, *) "  got      idx0=", Outcome - 1, " delta=", DeltaCoverage
  END IF
END DO
CLOSE(Unit)
WRITE(*, '(3A,I0,A)') "Checked ", TRIM(FileName), ": ", LineNo, " cases"
END SUBROUTINE CheckSampling

!==================================================================================================================================
!> Check every "area_density element_area xi expected_n" line against
!> MapAreaDensityToCoverage (model_area comes from the loaded model's own
!> ModelAreaNm2, not from the reference file -- see
!> generate_reference_data.py's dump_density_reference).
!==================================================================================================================================
SUBROUTINE CheckDensity(FileName, Model, NFail)
CHARACTER(LEN=*),      INTENT(IN)    :: FileName
TYPE(t_CoverageModel), INTENT(IN)    :: Model
INTEGER,               INTENT(INOUT) :: NFail
INTEGER :: Unit, IOStat, LineNo, ExpectedN, GotN
REAL(REAL64) :: AreaDensity, ElementArea, Xi
!----------------------------------------------------------------------------------------------------------------------------
OPEN(NEWUNIT=Unit, FILE=FileName, STATUS="OLD", ACTION="READ")
LineNo = 0
DO
  READ(Unit, *, IOSTAT=IOStat) AreaDensity, ElementArea, Xi, ExpectedN
  IF (IOStat /= 0) EXIT
  LineNo = LineNo + 1
  GotN = MapAreaDensityToCoverage(Model, AreaDensity, ElementArea, Xi)
  IF (GotN /= ExpectedN) THEN
    NFail = NFail + 1
    WRITE(*, '(3A,I0)') "MISMATCH in ", TRIM(FileName), " line ", LineNo
    WRITE(*, *) "  expected=", ExpectedN, " got=", GotN
  END IF
END DO
CLOSE(Unit)
WRITE(*, '(3A,I0,A)') "Checked ", TRIM(FileName), ": ", LineNo, " cases"
END SUBROUTINE CheckDensity

END PROGRAM test_coverage_kernel
