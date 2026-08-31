!==================================================================================================================================
!> Example integration program: loads the pooled and grid models exported
!> by examples/python/export_models_to_hdf5.py and computes the same
!> coverage-vs-(velocity,angle) heatmap examples/python/monte_carlo_walk.ipynb
!> does, but entirely from Fortran through MOD_CoverageKernel's public API --
!> this is meant to read like a worked example of what a DSMC boundary
!> condition's own per-particle loop looks like: own RNG stream (here,
!> Fortran's intrinsic RANDOM_NUMBER; a real host code would use its own),
!> own coverage state per surface element, calling SampleCoverageOutcome
!> once per step.
!>
!> Sweep (matches monte_carlo_walk.ipynb exactly): angles 0-80 deg in 10 deg
!> steps, velocities 1000-11000 m/s in 1000 m/s steps -- deliberately wider
!> than the sampled training grid (8 velocities x 5 angles, 2000-9319 m/s),
!> to also exercise the grid scheme's velocity extrapolation and the pooled
!> scheme's ceiling-grid interpolation outside their trusted range.
!>
!> For each (velocity, angle), runs a 20000-step random walk from
!> coverage=0 (coverage clipped at 0, matching the physical floor) and
!> reports the mean coverage over the last 8000 steps as that condition's
!> equilibrium estimate -- same N_STEPS/N_TAIL convention as the Python
!> notebook, though not a bit-for-bit RNG match (Fortran's own
!> RANDOM_NUMBER stream, not NumPy's).
!==================================================================================================================================
PROGRAM monte_carlo_walk
! MODULES
USE MOD_CoverageKernel
USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: REAL64
IMPLICIT NONE
!----------------------------------------------------------------------------------------------------------------------------
INTEGER, PARAMETER :: NAngles = 9                    !> 0, 10, ..., 80 deg
INTEGER, PARAMETER :: NVelocities = 11                !> 1000, 2000, ..., 11000 m/s
INTEGER, PARAMETER :: NSteps = 20000
INTEGER, PARAMETER :: NTail = 8000

REAL(wp) :: Angles(NAngles), Velocities(NVelocities)
TYPE(t_CoverageModel) :: PooledModel, GridModel
INTEGER :: i, SeedSize
INTEGER, ALLOCATABLE :: Seed(:)
!==================================================================================================================================
! Fixed seed for a reproducible example run (not a bit-for-bit match to
! monte_carlo_walk.ipynb's NumPy RNG -- just the same "seed=0" spirit).
CALL RANDOM_SEED(SIZE=SeedSize)
ALLOCATE(Seed(SeedSize))
Seed = 0
CALL RANDOM_SEED(PUT=Seed)

DO i = 1, NAngles
  Angles(i) = REAL((i - 1) * 10, wp)
END DO
DO i = 1, NVelocities
  Velocities(i) = REAL(1000 * i, wp)
END DO

CALL LoadCoverageModel("../python/models/pooled_model.h5", PooledModel)
CALL LoadCoverageModel("../python/models/grid_model.h5", GridModel)

CALL RunSweep("heatmap_pooled.csv", PooledModel, Velocities, Angles)
CALL RunSweep("heatmap_grid.csv", GridModel, Velocities, Angles)

WRITE(*, *) "Wrote heatmap_pooled.csv and heatmap_grid.csv"

CONTAINS

!==================================================================================================================================
!> Run the full (velocity, angle) sweep for one loaded model and write
!> "velocity,angle,mean_coverage" rows to FileName.
!==================================================================================================================================
SUBROUTINE RunSweep(FileName, Model, Velocities, Angles)
CHARACTER(LEN=*),      INTENT(IN) :: FileName
TYPE(t_CoverageModel), INTENT(IN) :: Model
REAL(wp),              INTENT(IN) :: Velocities(:), Angles(:)
INTEGER :: Unit, vi, ai
REAL(wp) :: MeanCoverage
!----------------------------------------------------------------------------------------------------------------------------
OPEN(NEWUNIT=Unit, FILE=FileName, STATUS="REPLACE", ACTION="WRITE")
WRITE(Unit, '(A)') "velocity,angle,mean_coverage"
DO vi = 1, SIZE(Velocities)
  DO ai = 1, SIZE(Angles)
    MeanCoverage = RunOneWalk(Model, Velocities(vi), Angles(ai))
    WRITE(Unit, '(F10.2,A,F8.2,A,F12.6)') Velocities(vi), ",", Angles(ai), ",", MeanCoverage
  END DO
END DO
CLOSE(Unit)
END SUBROUTINE RunSweep

!==================================================================================================================================
!> One condition's random walk: NSteps steps starting at coverage=0,
!> coverage clipped at 0 after every step (the physical floor -- coverage
!> can't go negative), returns the mean coverage over the last NTail steps.
!==================================================================================================================================
FUNCTION RunOneWalk(Model, Velocity, Angle) RESULT(MeanCoverage)
TYPE(t_CoverageModel), INTENT(IN) :: Model
REAL(wp),              INTENT(IN) :: Velocity, Angle
REAL(wp) :: MeanCoverage
REAL(wp) :: Coverage, Xi, DeltaCoverage, Probs(Model%NClasses)
REAL(wp) :: TailSum
INTEGER  :: Step, Outcome
!----------------------------------------------------------------------------------------------------------------------------
Coverage = 0.0_wp
TailSum = 0.0_wp
DO Step = 1, NSteps
  CALL RANDOM_NUMBER(Xi)
  CALL SampleCoverageOutcome(Model, Coverage, Angle, Velocity, Xi, Outcome, DeltaCoverage, Probs)
  Coverage = MAX(Coverage + DeltaCoverage, 0.0_wp)
  IF (Step > NSteps - NTail) TailSum = TailSum + Coverage
END DO
MeanCoverage = TailSum / REAL(NTail, wp)
END FUNCTION RunOneWalk

END PROGRAM monte_carlo_walk
