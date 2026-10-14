!==================================================================================================================================
!> Worked integration example: the coverage walk a DSMC boundary condition
!> runs, through MOD_CoverageKernel's public API only.
!>
!> Loads the paper's coverage-weighted NN ensemble and grid model and, for
!> the same velocity/angle sweep as examples/python/coverage_walk.py,
!> computes
!>   - the exact long-run coverage average (stationary distribution of the
!>     walk on the integer coverages 0..NSites: pi(c+1)/pi(c) = P_up(c)/P_down(c+1)),
!>   - the average of a Monte Carlo walk: one SampleCoverageOutcome call per
!>     impact with the host's own random numbers (here RANDOM_NUMBER),
!> and writes walk_sweep_fortran.csv. `python ../python/coverage_walk.py
!> --compare` checks it against the Python results.
!>
!> The host owns the random-number stream and the coverage state of every
!> surface element; the model is read-only after loading.
!>
!> Run from examples/fortran/:   ./build/monte_carlo_walk [nn_file grid_file]
!==================================================================================================================================
PROGRAM monte_carlo_walk
! MODULES
USE MOD_CoverageKernel
IMPLICIT NONE
!----------------------------------------------------------------------------------------------------------------------------
INTEGER,  PARAMETER :: NSteps = 20000, NTail = 8000
REAL(wp), PARAMETER :: Velocities(3) = [5000.0_wp, 7923.9_wp, 9319.0_wp]
REAL(wp), PARAMETER :: Angles(9) = [0.0_wp, 10.0_wp, 20.0_wp, 30.0_wp, 40.0_wp, 50.0_wp, 60.0_wp, 70.0_wp, 80.0_wp]
CHARACTER(LEN=256) :: Files(2)
CHARACTER(LEN=16), PARAMETER :: Names(2) = ["nn_ensemble     ", "grid            "]
TYPE(t_CoverageModel) :: Model
INTEGER :: m, iv, ia, Unit, SeedSize
INTEGER, ALLOCATABLE :: Seed(:)
REAL(wp) :: Stationary, WalkAvg
!==================================================================================================================================
Files(1) = "../data/models/branching_nn_ensemble_weighted.h5"
Files(2) = "../data/models/grid_model.h5"
IF (COMMAND_ARGUMENT_COUNT() == 2) THEN
  CALL GET_COMMAND_ARGUMENT(1, Files(1))
  CALL GET_COMMAND_ARGUMENT(2, Files(2))
END IF

CALL RANDOM_SEED(SIZE=SeedSize)
ALLOCATE(Seed(SeedSize))
Seed = 12345
CALL RANDOM_SEED(PUT=Seed)

OPEN(NEWUNIT=Unit, FILE="walk_sweep_fortran.csv", STATUS="REPLACE", ACTION="WRITE")
WRITE(Unit, '(A)') "model,velocity,angle,stationary_average,walk_average"
DO m = 1, 2
  CALL LoadCoverageModel(TRIM(Files(m)), Model)
  DO iv = 1, SIZE(Velocities)
    DO ia = 1, SIZE(Angles)
      Stationary = StationaryAverage(Model, Velocities(iv), Angles(ia))
      WalkAvg = WalkAverage(Model, Velocities(iv), Angles(ia))
      WRITE(Unit, '(A,",",F0.1,",",F0.1,",",ES23.16,",",ES23.16)') TRIM(Names(m)), Velocities(iv), Angles(ia), &
            Stationary, WalkAvg
    END DO
  END DO
  WRITE(*, '(3A)') "done: ", TRIM(Names(m)), " model"
END DO
CLOSE(Unit)
WRITE(*, '(A)') "wrote walk_sweep_fortran.csv"

CONTAINS

!> Index of the class with coverage change Delta (+1 or -1).
INTEGER FUNCTION ClassWithDelta(Model, Delta)
TYPE(t_CoverageModel), INTENT(IN) :: Model
REAL(wp),              INTENT(IN) :: Delta
INTEGER :: k
ClassWithDelta = 0
DO k = 1, Model%NClasses
  IF (Model%CoverageDelta(k) == Delta) ClassWithDelta = k
END DO
IF (ClassWithDelta == 0) ERROR STOP "monte_carlo_walk: model needs one class with coverage change +1 and one with -1"
END FUNCTION ClassWithDelta

!> Exact long-run mean coverage of the walk on the integer coverages 0..NSites.
REAL(wp) FUNCTION StationaryAverage(Model, Velocity, Angle)
TYPE(t_CoverageModel), INTENT(IN) :: Model
REAL(wp),              INTENT(IN) :: Velocity, Angle
REAL(wp) :: P(Model%NClasses, 0:Model%NSites), LogPi(0:Model%NSites), Pi(0:Model%NSites)
INTEGER :: c, Up, Down, Last
Up = ClassWithDelta(Model, 1.0_wp)
Down = ClassWithDelta(Model, -1.0_wp)
DO c = 0, Model%NSites
  CALL EvalCoverageProbs(Model, REAL(c, wp), Velocity, Angle, P(:, c))
END DO
LogPi = 0.0_wp
Last = Model%NSites            ! highest coverage the walk can reach
DO c = 0, Model%NSites - 1
  IF (P(Up, c) <= 0.0_wp .OR. P(Down, c + 1) <= 0.0_wp) THEN
    Last = c
    EXIT
  END IF
  LogPi(c + 1) = LogPi(c) + LOG(P(Up, c)) - LOG(P(Down, c + 1))
END DO
Pi = 0.0_wp
Pi(0:Last) = EXP(LogPi(0:Last) - MAXVAL(LogPi(0:Last)))
Pi = Pi / SUM(Pi)
StationaryAverage = SUM(Pi * [(REAL(c, wp), c = 0, Model%NSites)])
END FUNCTION StationaryAverage

!> Mean coverage over the last NTail of NSteps impacts, starting from coverage 0.
REAL(wp) FUNCTION WalkAverage(Model, Velocity, Angle)
TYPE(t_CoverageModel), INTENT(IN) :: Model
REAL(wp),              INTENT(IN) :: Velocity, Angle
REAL(wp) :: Coverage, Xi, Delta, Probs(Model%NClasses), Total
INTEGER :: Step, Outcome
Coverage = 0.0_wp
Total = 0.0_wp
DO Step = 1, NSteps
  CALL RANDOM_NUMBER(Xi)
  CALL SampleCoverageOutcome(Model, Coverage, Velocity, Angle, Xi, Outcome, Delta, Probs)
  Coverage = MAX(0.0_wp, Coverage + Delta)
  IF (Step > NSteps - NTail) Total = Total + Coverage
END DO
WalkAverage = Total / REAL(NTail, wp)
END FUNCTION WalkAverage

END PROGRAM monte_carlo_walk
