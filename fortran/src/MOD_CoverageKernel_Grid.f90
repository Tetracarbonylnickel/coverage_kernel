!==================================================================================================================================
!> GENERIC: recursive local-stencil tensor-product grid evaluation.
!> Mirrors grid_classifier.py's GridClassifier.eval: this module has no
!> idea what the grid axes or the per-cell probability vector mean -- it
!> doesn't know about velocity, angle, coverage, or the coverage blend
!> physics. It only knows how to (a) resolve, per axis, which local window
!> of grid indices and which effective 1-D method (honoring each axis's own
!> edge policy) a query needs, and (b) recursively combine whatever a
!> caller-supplied CellEval callback returns at each leaf cell.
!>
!> The physics of *what* CellEval computes (the coverage blend/gate, using
!> each cell's own CoverageMax/Ceiling) is supplied by the domain layer
!> (MOD_CoverageKernel), exactly the way grid_classifier.py's
!> GridClassifier.eval calls whatever .eval(feature) the object stored in
!> cell_classifiers happens to implement (there, a CoverageCell; here, an
!> internal procedure passed in as a PROCEDURE dummy argument) -- keeping
!> this module reusable for a grid of classifiers over any other feature
!> entirely.
!>
!> Evaluating one query only ever touches the grid cells inside the local
!> interpolation stencil around it (<=2 points/axis for "linear", <=4 for
!> "pchip") -- never the whole grid. Tensor-product interpolation across
!> NAxes dimensions is done by recursively collapsing one axis at a time.
!==================================================================================================================================
MODULE MOD_CoverageKernel_Grid
! MODULES
USE MOD_CoverageKernel_Types,    ONLY: wp, t_GridAxis
USE MOD_CoverageKernel_Interp1D, ONLY: LocalWindow, LinearEvalWindowVec, PchipEvalWindowVec, FlatIndexCOrder
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
ABSTRACT INTERFACE
  !> Evaluate one grid cell's own (already blended, in the domain layer's
  !> sense) probability vector at the given per-cell feature value.
  FUNCTION CellEvalInterface(FlatIdx, Feature, NClasses) RESULT(Probs)
    IMPORT :: wp
    INTEGER,  INTENT(IN) :: FlatIdx, NClasses
    REAL(wp), INTENT(IN) :: Feature
    REAL(wp) :: Probs(NClasses)
  END FUNCTION CellEvalInterface
END INTERFACE

PUBLIC :: CellEvalInterface
PUBLIC :: EvalGridModel
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> Top-level entry point: evaluate the whole grid at AxisQuery(1:NAxes)
!> (e.g. [velocity, angle]) for per-cell feature value Feature (e.g.
!> coverage), combining NClasses-length cell probability vectors produced
!> by CellEval. Clips the tensor-product result to >=0 and renormalizes as
!> a safety net (pchip has no structural simplex guarantee the way a
!> convex-combination method like "linear" does, though pchip was chosen
!> for empirically negligible overshoot -- see grid_classifier.py).
!==================================================================================================================================
FUNCTION EvalGridModel(Axes, Shape_, Method, Feature, AxisQuery, CellEval, NClasses) RESULT(Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_GridAxis), INTENT(IN) :: Axes(:)
INTEGER,          INTENT(IN) :: Shape_(:)
CHARACTER(LEN=*), INTENT(IN) :: Method
REAL(wp),         INTENT(IN) :: Feature
REAL(wp),         INTENT(IN) :: AxisQuery(:)
PROCEDURE(CellEvalInterface)  :: CellEval
INTEGER,          INTENT(IN) :: NClasses
REAL(wp) :: Probs(NClasses)
! LOCAL VARIABLES
REAL(wp) :: RawTotal
INTEGER  :: EmptyIdx(0)
!----------------------------------------------------------------------------------------------------------------------------
Probs = EvalGridRecursive(Axes, Shape_, Method, Feature, AxisQuery, CellEval, NClasses, 1, EmptyIdx)
Probs = MAX(Probs, 0.0_wp)
RawTotal = SUM(Probs)
IF (RawTotal > 0.0_wp) Probs = Probs / RawTotal
END FUNCTION EvalGridModel

!==================================================================================================================================
!> Recursively collapse axis Depth..NAxes: at the leaf (Depth > NAxes),
!> evaluate the cell at FixedIdx via CellEval; otherwise resolve this
!> axis's local window/effective method (honoring its own edge policy),
!> recurse for every window index, and interpolate the results along this
!> axis.
!==================================================================================================================================
RECURSIVE FUNCTION EvalGridRecursive(Axes, Shape_, Method, Feature, AxisQuery, CellEval, NClasses, Depth, FixedIdx) &
    RESULT(Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_GridAxis), INTENT(IN) :: Axes(:)
INTEGER,          INTENT(IN) :: Shape_(:)
CHARACTER(LEN=*), INTENT(IN) :: Method
REAL(wp),         INTENT(IN) :: Feature
REAL(wp),         INTENT(IN) :: AxisQuery(:)
PROCEDURE(CellEvalInterface)  :: CellEval
INTEGER,          INTENT(IN) :: NClasses
INTEGER,          INTENT(IN) :: Depth
INTEGER,          INTENT(IN) :: FixedIdx(:)
REAL(wp) :: Probs(NClasses)
! LOCAL VARIABLES
INTEGER  :: NAxes, Lo, Hi, WindowSize, w, FlatIdx
REAL(wp) :: Grid1D(SIZE(Axes(MIN(Depth, SIZE(Axes)))%Values))
REAL(wp) :: Qv, AxLo, AxHi
CHARACTER(LEN=32) :: EffMethod
REAL(wp), ALLOCATABLE :: Values(:,:), XWindow(:)
LOGICAL :: IsLeftEdge, IsRightEdge
!----------------------------------------------------------------------------------------------------------------------------
NAxes = SIZE(Shape_)
IF (Depth > NAxes) THEN
  FlatIdx = FlatIndexCOrder(Shape_, FixedIdx)
  Probs = CellEval(FlatIdx, Feature, NClasses)
  RETURN
END IF

Grid1D = Axes(Depth)%Values
AxLo = Grid1D(1)
AxHi = Grid1D(SIZE(Grid1D))

SELECT CASE (TRIM(Axes(Depth)%EdgePolicy))
CASE ("clip")
  Qv = MIN(MAX(AxisQuery(Depth), AxLo), AxHi)
  EffMethod = Method
CASE ("linear_extrapolate")
  Qv = AxisQuery(Depth)
  IF (Qv >= AxLo .AND. Qv <= AxHi) THEN
    EffMethod = Method
  ELSE
    EffMethod = "linear"
  END IF
CASE DEFAULT
  ERROR STOP "MOD_CoverageKernel_Grid: unknown axis edge_policy"
END SELECT

IF (SIZE(Grid1D) == 1) THEN
  Probs = EvalGridRecursive(Axes, Shape_, Method, Feature, AxisQuery, CellEval, NClasses, Depth + 1, [FixedIdx, 1])
  RETURN
END IF

CALL LocalWindow(Grid1D, Qv, TRIM(EffMethod), Lo, Hi)
WindowSize = Hi - Lo + 1
ALLOCATE(Values(WindowSize, NClasses))
DO w = 1, WindowSize
  Values(w, :) = EvalGridRecursive(Axes, Shape_, Method, Feature, AxisQuery, CellEval, NClasses, &
                                    Depth + 1, [FixedIdx, Lo + w - 1])
END DO
XWindow = Grid1D(Lo:Hi)

IF (TRIM(EffMethod) == "linear") THEN
  Probs = LinearEvalWindowVec(XWindow(1), XWindow(WindowSize), Values(1, :), Values(WindowSize, :), Qv)
ELSE
  IsLeftEdge  = (Lo == 1)
  IsRightEdge = (Hi == SIZE(Grid1D))
  Probs = PchipEvalWindowVec(XWindow, Values, Qv, IsLeftEdge, IsRightEdge)
END IF
END FUNCTION EvalGridRecursive

END MODULE MOD_CoverageKernel_Grid
