!==================================================================================================================================
!> Cross-validation harness: loads the HDF5 models written by
!> generate_reference_data.py and checks that MOD_CoverageKernel's public
!> API reproduces the Python reference numbers (within Tol) for a battery of
!> queries, including out-of-range velocities and angles, coverage 0 and the
!> blend region beyond the sampled coverage range. Models:
!>   grid_test.h5, nn_test.h5   small synthetic models
!>   grid_real.h5, nn_real.h5   the grid model and the coverage-weighted NN
!>                              ensemble of the paper (if present)
!> Also checks SampleCoverageOutcome, MapAdsorbateCountToCoverage and the
!> IErr code for a missing file. Every model is loaded in turn, which also
!> checks that loading leaves the HDF5 library usable.
!==================================================================================================================================
PROGRAM test_coverage_kernel
! MODULES
USE MOD_CoverageKernel
USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: REAL64
IMPLICIT NONE
!----------------------------------------------------------------------------------------------------------------------------
REAL(REAL64), PARAMETER :: Tol = 1.0E-9_REAL64
TYPE(t_CoverageModel) :: Model
INTEGER :: NFail, IErr
LOGICAL :: HaveReal
!==================================================================================================================================
NFail = 0

CALL LoadCoverageModel("data/does_not_exist.h5", Model, IErr)
IF (IErr /= 1) THEN
  NFail = NFail + 1
  WRITE(*, '(A,I0)') "MISMATCH: missing file should give IErr = 1, got ", IErr
ELSE
  WRITE(*, *) "Checked missing-file IErr"
END IF

CALL LoadCoverageModel("data/grid_test.h5", Model)
CALL CheckProbsFile("data/reference_grid_test.txt", Model, NFail)

CALL LoadCoverageModel("data/nn_test.h5", Model)
CALL CheckProbsFile("data/reference_nn_test.txt", Model, NFail)
CALL CheckSampling("data/reference_sampling.txt", Model, NFail)
CALL CheckCount("data/reference_count.txt", Model, NFail)

INQUIRE(FILE="data/nn_real.h5", EXIST=HaveReal)
IF (HaveReal) THEN
  CALL LoadCoverageModel("data/grid_real.h5", Model)
  CALL CheckProbsFile("data/reference_grid_real.txt", Model, NFail)
  CALL LoadCoverageModel("data/nn_real.h5", Model)
  CALL CheckProbsFile("data/reference_nn_real.txt", Model, NFail)
ELSE
  WRITE(*, *) "Skipping the real models: data/nn_real.h5 not found"
END IF

IF (NFail > 0) THEN
  WRITE(*, '(A,I0,A)') "FAILED: ", NFail, " mismatch(es)"
  ERROR STOP "coverage_kernel Fortran-vs-Python cross-validation FAILED"
ELSE
  WRITE(*, *) "All coverage_kernel Fortran-vs-Python cross-validation checks PASSED"
END IF

CONTAINS

!==================================================================================================================================
!> Check every "coverage velocity angle p_1 .. p_K" line of a reference file
!> against EvalCoverageProbs(Model, coverage, velocity, angle, ...).
!==================================================================================================================================
SUBROUTINE CheckProbsFile(FileName, Model, NFail)
CHARACTER(LEN=*),      INTENT(IN)    :: FileName
TYPE(t_CoverageModel), INTENT(IN)    :: Model
INTEGER,               INTENT(INOUT) :: NFail
INTEGER :: Unit, IOStat, LineNo
REAL(REAL64) :: Coverage, Velocity, Angle, MaxErr, WorstErr
REAL(REAL64) :: Expected(Model%NClasses), Probs(Model%NClasses)
!----------------------------------------------------------------------------------------------------------------------------
OPEN(NEWUNIT=Unit, FILE=FileName, STATUS="OLD", ACTION="READ")
LineNo = 0
WorstErr = 0.0_REAL64
DO
  READ(Unit, *, IOSTAT=IOStat) Coverage, Velocity, Angle, Expected
  IF (IOStat /= 0) EXIT
  LineNo = LineNo + 1
  CALL EvalCoverageProbs(Model, Coverage, Velocity, Angle, Probs)
  MaxErr = MAXVAL(ABS(Probs - Expected))
  WorstErr = MAX(WorstErr, MaxErr)
  IF (MaxErr > Tol) THEN
    NFail = NFail + 1
    WRITE(*, '(3A,I0,A,ES12.4)') "MISMATCH in ", TRIM(FileName), " line ", LineNo, ": max abs err = ", MaxErr
    WRITE(*, *) "  coverage=", Coverage, " velocity=", Velocity, " angle=", Angle
    WRITE(*, *) "  expected:", Expected
    WRITE(*, *) "  got:     ", Probs
  END IF
END DO
CLOSE(Unit)
WRITE(*, '(3A,I0,A,ES10.2)') "Checked ", TRIM(FileName), ": ", LineNo, " cases, largest difference ", WorstErr
END SUBROUTINE CheckProbsFile

!==================================================================================================================================
!> Check every "coverage velocity angle xi expected_idx0 expected_delta" line
!> against SampleCoverageOutcome (1-based Outcome compared as 0-based).
!==================================================================================================================================
SUBROUTINE CheckSampling(FileName, Model, NFail)
CHARACTER(LEN=*),      INTENT(IN)    :: FileName
TYPE(t_CoverageModel), INTENT(IN)    :: Model
INTEGER,               INTENT(INOUT) :: NFail
INTEGER :: Unit, IOStat, LineNo, ExpectedIdx0, Outcome
REAL(REAL64) :: Coverage, Velocity, Angle, Xi, ExpectedDelta, DeltaCoverage
REAL(REAL64) :: Probs(Model%NClasses)
!----------------------------------------------------------------------------------------------------------------------------
OPEN(NEWUNIT=Unit, FILE=FileName, STATUS="OLD", ACTION="READ")
LineNo = 0
DO
  READ(Unit, *, IOSTAT=IOStat) Coverage, Velocity, Angle, Xi, ExpectedIdx0, ExpectedDelta
  IF (IOStat /= 0) EXIT
  LineNo = LineNo + 1
  CALL SampleCoverageOutcome(Model, Coverage, Velocity, Angle, Xi, Outcome, DeltaCoverage, Probs)
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
!> Check every "adsorbate_count element_area xi expected_coverage" line
!> against MapAdsorbateCountToCoverage (the model area comes from the file).
!==================================================================================================================================
SUBROUTINE CheckCount(FileName, Model, NFail)
CHARACTER(LEN=*),      INTENT(IN)    :: FileName
TYPE(t_CoverageModel), INTENT(IN)    :: Model
INTEGER,               INTENT(INOUT) :: NFail
INTEGER :: Unit, IOStat, LineNo, ExpectedN, GotN
REAL(REAL64) :: NAds, ElementArea, Xi
!----------------------------------------------------------------------------------------------------------------------------
OPEN(NEWUNIT=Unit, FILE=FileName, STATUS="OLD", ACTION="READ")
LineNo = 0
DO
  READ(Unit, *, IOSTAT=IOStat) NAds, ElementArea, Xi, ExpectedN
  IF (IOStat /= 0) EXIT
  LineNo = LineNo + 1
  GotN = MapAdsorbateCountToCoverage(Model, NAds, ElementArea, Xi)
  IF (GotN /= ExpectedN) THEN
    NFail = NFail + 1
    WRITE(*, '(3A,I0)') "MISMATCH in ", TRIM(FileName), " line ", LineNo
    WRITE(*, *) "  expected=", ExpectedN, " got=", GotN
  END IF
END DO
CLOSE(Unit)
WRITE(*, '(3A,I0,A)') "Checked ", TRIM(FileName), ": ", LineNo, " cases"
END SUBROUTINE CheckCount

END PROGRAM test_coverage_kernel
