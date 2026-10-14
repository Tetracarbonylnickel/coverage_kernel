!==================================================================================================================================
!> GENERIC: local-window 1-D interpolation building blocks, reused by
!> MOD_CoverageKernel_Grid to interpolate across a sampled grid axis (e.g.
!> velocity or angle). Nothing here knows about coverage, outcomes, or any
!> other domain concept -- it operates on plain (x, y) arrays. Mirrors
!> interp1d.py exactly (see that file's docstring for the numerical
!> background on why PCHIP is evaluated from a local window rather than a
!> full tensor-product spline).
!>
!> Two 1-D methods, both from a local window of grid points rather than the
!> whole axis: "linear" (2 points) and "pchip" (up to 4 points, fewer at
!> the ends of the axis). Vector-valued Y (one value per outcome class) is
!> supported throughout since that's the only use this project has for it
!> (interpolating a whole probability vector across a grid axis).
!==================================================================================================================================
MODULE MOD_CoverageKernel_Interp1D
! MODULES
USE MOD_CoverageKernel_Types, ONLY: wp
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
PUBLIC :: LocalWindow
PUBLIC :: LinearEvalWindowVec
PUBLIC :: PchipEvalWindowVec
PUBLIC :: MultilinearClipped
PUBLIC :: FlatIndexCOrder
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> 1-based index of the last grid point <= X (i.e. NumPy's
!> searchsorted(grid, x, side="right") translated directly to 1-based
!> indexing), unclamped. Internal helper shared by LocalWindow and the
!> pchip bracket search inside PchipEvalWindowVec.
!==================================================================================================================================
PURE FUNCTION LastIndexLE(Grid, X) RESULT(I)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: Grid(:)
REAL(wp), INTENT(IN) :: X
INTEGER :: I
! LOCAL VARIABLES
INTEGER :: k
!----------------------------------------------------------------------------------------------------------------------------
I = 1
DO k = 1, SIZE(Grid)
  IF (Grid(k) <= X) THEN
    I = k
  ELSE
    EXIT
  END IF
END DO
END FUNCTION LastIndexLE

!==================================================================================================================================
!> Return the [Lo, Hi] inclusive 1-based index range of grid points needed
!> to evaluate Method at query X. "linear" needs only the bracketing pair;
!> "pchip" needs up to 4 points {i-1,i,i+1,i+2} (fewer near either edge of
!> the axis -- see PchipEvalWindowVec's edge-derivative handling).
!==================================================================================================================================
SUBROUTINE LocalWindow(Grid, X, Method, Lo, Hi)
! INPUT / OUTPUT VARIABLES
REAL(wp),         INTENT(IN)  :: Grid(:)
REAL(wp),         INTENT(IN)  :: X
CHARACTER(LEN=*), INTENT(IN)  :: Method
INTEGER,          INTENT(OUT) :: Lo, Hi
! LOCAL VARIABLES
INTEGER :: n, i
!----------------------------------------------------------------------------------------------------------------------------
n = SIZE(Grid)
i = MAX(1, MIN(LastIndexLE(Grid, X), n - 1))    ! bracket: Grid(i) <= X <= Grid(i+1), clamped to a valid interval
SELECT CASE (TRIM(Method))
CASE ("linear")
  Lo = i
  Hi = i + 1
CASE ("pchip")
  Lo = MAX(i - 1, 1)
  Hi = MIN(i + 2, n)
CASE DEFAULT
  ERROR STOP "MOD_CoverageKernel_Interp1D: unknown interpolation method"
END SELECT
END SUBROUTINE LocalWindow

!==================================================================================================================================
!> Elementwise sign, matching NumPy's np.sign (-1, 0, or +1) -- distinct
!> from Fortran's intrinsic SIGN(A,B), which returns |A| with B's sign.
!==================================================================================================================================
ELEMENTAL FUNCTION SignOf(X) RESULT(S)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: X
INTEGER :: S
!----------------------------------------------------------------------------------------------------------------------------
S = 0
IF (X > 0.0_wp) S = 1
IF (X < 0.0_wp) S = -1
END FUNCTION SignOf

!==================================================================================================================================
!> One-sided 3-point derivative estimate at an axis endpoint
!> (scipy.interpolate._cubic._edge_case), elementwise over a vector Y.
!> H0/M0 are the step/slope of the interval touching the endpoint; H1/M1
!> are the next interval inward.
!==================================================================================================================================
FUNCTION EdgeDerivative(H0, H1, M0, M1) RESULT(D)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: H0, H1
REAL(wp), INTENT(IN) :: M0(:), M1(:)
REAL(wp) :: D(SIZE(M0))
! LOCAL VARIABLES
LOGICAL :: Override(SIZE(M0))
!----------------------------------------------------------------------------------------------------------------------------
D = ((2.0_wp * H0 + H1) * M0 - H0 * M1) / (H0 + H1)
WHERE (SignOf(D) /= SignOf(M0)) D = 0.0_wp
Override = (SignOf(M0) /= SignOf(M1)) .AND. (ABS(D) > 3.0_wp * ABS(M0))
WHERE (Override) D = 3.0_wp * M0
END FUNCTION EdgeDerivative

!==================================================================================================================================
!> Fritsch-Carlson weighted-harmonic-mean derivative estimate at an
!> interior node, elementwise over a vector Y, given the two secant slopes
!> either side of it (scipy.interpolate._cubic._find_derivatives' interior
!> branch).
!==================================================================================================================================
FUNCTION InteriorDerivative(HPrev, HNext, MPrev, MNext) RESULT(D)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: HPrev, HNext
REAL(wp), INTENT(IN) :: MPrev(:), MNext(:)
REAL(wp) :: D(SIZE(MPrev))
! LOCAL VARIABLES
LOGICAL  :: Condition(SIZE(MPrev))
REAL(wp) :: W1, W2, MPrevSafe(SIZE(MPrev)), MNextSafe(SIZE(MPrev)), WhMean(SIZE(MPrev))
!----------------------------------------------------------------------------------------------------------------------------
Condition = (SignOf(MPrev) /= SignOf(MNext)) .OR. (MPrev == 0.0_wp) .OR. (MNext == 0.0_wp)
W1 = 2.0_wp * HNext + HPrev
W2 = HNext + 2.0_wp * HPrev
MPrevSafe = MERGE(1.0_wp, MPrev, MPrev == 0.0_wp)
MNextSafe = MERGE(1.0_wp, MNext, MNext == 0.0_wp)
WhMean = (W1 / MPrevSafe + W2 / MNextSafe) / (W1 + W2)
D = MERGE(0.0_wp, 1.0_wp / WhMean, Condition)
END FUNCTION InteriorDerivative

!==================================================================================================================================
!> Ordinary linear interpolation between the two bracketing points X0/X1
!> (Y0/Y1 may be per-outcome vectors).
!==================================================================================================================================
FUNCTION LinearEvalWindowVec(X0, X1, Y0, Y1, XQuery) RESULT(Y)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: X0, X1
REAL(wp), INTENT(IN) :: Y0(:), Y1(:)
REAL(wp), INTENT(IN) :: XQuery
REAL(wp) :: Y(SIZE(Y0))
! LOCAL VARIABLES
REAL(wp) :: T
!----------------------------------------------------------------------------------------------------------------------------
IF (X1 == X0) THEN
  Y = Y0
ELSE
  T = (XQuery - X0) / (X1 - X0)
  Y = Y0 + T * (Y1 - Y0)
END IF
END FUNCTION LinearEvalWindowVec

!==================================================================================================================================
!> Evaluate PCHIP (Fritsch-Carlson monotone cubic Hermite) at XQuery using
!> only the local window XWindow(1:N)/YWindow(1:N,:) reported by
!> LocalWindow (N = 2, 3, or 4; fewer only if the whole axis has < 4
!> points). IsLeftEdge/IsRightEdge report whether XWindow(1)/XWindow(N) is
!> the *global* first/last point of the full axis (not just this local
!> window) -- this selects the one-sided edge derivative formula there
!> instead of the interior weighted-harmonic-mean formula, exactly
!> mirroring what evaluating the full axis with
!> scipy.interpolate.PchipInterpolator would do.
!>
!> Precondition (guaranteed by how MOD_CoverageKernel_Grid constructs
!> XWindow/YWindow via LocalWindow, never violated through that call path):
!> the bracket found within the window can only fall at its very first
!> sub-interval if XWindow(1) is also the true global edge.
!==================================================================================================================================
FUNCTION PchipEvalWindowVec(XWindow, YWindow, XQuery, IsLeftEdge, IsRightEdge) RESULT(Y)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: XWindow(:)
REAL(wp), INTENT(IN) :: YWindow(:,:)     !> (SIZE(XWindow), NClasses)
REAL(wp), INTENT(IN) :: XQuery
LOGICAL,  INTENT(IN) :: IsLeftEdge, IsRightEdge
REAL(wp) :: Y(SIZE(YWindow, 2))
! LOCAL VARIABLES
INTEGER  :: n, K, i0, i1, k_
REAL(wp) :: H(SIZE(XWindow) - 1)
REAL(wp) :: M(SIZE(XWindow) - 1, SIZE(YWindow, 2))
REAL(wp) :: D0(SIZE(YWindow, 2)), D1(SIZE(YWindow, 2))
REAL(wp) :: X0, X1, Hb, T, H00, H10, H01, H11
!----------------------------------------------------------------------------------------------------------------------------
n = SIZE(XWindow)
K = SIZE(YWindow, 2)

DO k_ = 1, n - 1
  H(k_) = XWindow(k_ + 1) - XWindow(k_)
  M(k_, :) = (YWindow(k_ + 1, :) - YWindow(k_, :)) / H(k_)
END DO

IF (n == 2) THEN
  D0 = M(1, :)
  D1 = M(1, :)
  i0 = 1
  i1 = 2
ELSE
  i0 = MAX(1, MIN(LastIndexLE(XWindow, XQuery), n - 1))
  i1 = i0 + 1

  IF (i0 == 1 .AND. IsLeftEdge) THEN
    D0 = EdgeDerivative(H(1), H(2), M(1, :), M(2, :))
  ELSE IF (i0 == n .AND. IsRightEdge) THEN
    D0 = EdgeDerivative(H(n - 1), H(n - 2), M(n - 1, :), M(n - 2, :))
  ELSE
    D0 = InteriorDerivative(H(i0 - 1), H(i0), M(i0 - 1, :), M(i0, :))
  END IF

  IF (i1 == n .AND. IsRightEdge) THEN
    D1 = EdgeDerivative(H(n - 1), H(n - 2), M(n - 1, :), M(n - 2, :))
  ELSE
    D1 = InteriorDerivative(H(i1 - 1), H(i1), M(i1 - 1, :), M(i1, :))
  END IF
END IF

X0 = XWindow(i0); X1 = XWindow(i1)
Hb = X1 - X0
T = (XQuery - X0) / Hb

H00 =  2.0_wp * T**3 - 3.0_wp * T**2 + 1.0_wp
H10 =         T**3 - 2.0_wp * T**2 + T
H01 = -2.0_wp * T**3 + 3.0_wp * T**2
H11 =         T**3 -          T**2

Y = H00 * YWindow(i0, :) + H10 * Hb * D0 + H01 * YWindow(i1, :) + H11 * Hb * D1
END FUNCTION PchipEvalWindowVec

!==================================================================================================================================
!> N-dimensional multilinear interpolation of a small precomputed flattened
!> array Values (C-order / last-axis-fastest, matching how
!> MOD_CoverageKernel_IO reads it from HDF5 and how
!> MOD_CoverageKernel_Grid's FlatIndex computes offsets), clipping the
!> query into range on every axis first (never extrapolates). Used for the
!> NN ensemble's ceiling grid, where the interpolated *value* is a smooth
!> safety boundary (always linear, never pchip -- no shape-preservation
!> concern, and the grid is cheap enough that lazy evaluation buys nothing
!> here unlike the per-cell classifiers in MOD_CoverageKernel_Grid).
!==================================================================================================================================
RECURSIVE FUNCTION MultilinearClipped(Axes, AxesShape, Values, Query, Depth, FixedIdx) RESULT(V)
! MODULES
USE MOD_CoverageKernel_Types, ONLY: t_GridAxis
! INPUT / OUTPUT VARIABLES
TYPE(t_GridAxis), INTENT(IN) :: Axes(:)
INTEGER,          INTENT(IN) :: AxesShape(:)
REAL(wp),         INTENT(IN) :: Values(:)          !> flattened, C-order, length PRODUCT(AxesShape)
REAL(wp),         INTENT(IN) :: Query(:)            !> (SIZE(Axes))
INTEGER,          INTENT(IN) :: Depth                !> current axis, 1-based; call with Depth=1
INTEGER,          INTENT(IN) :: FixedIdx(:)          !> indices fixed for axes 1..Depth-1 (1-based); call with size 0
REAL(wp) :: V
! LOCAL VARIABLES
INTEGER  :: NAxes, Lo, Hi, Flat
REAL(wp), ALLOCATABLE :: Grid1D(:)
REAL(wp) :: Qv, VLo, VHi, T
!----------------------------------------------------------------------------------------------------------------------------
NAxes = SIZE(Axes)
IF (Depth > NAxes) THEN
  Flat = FlatIndexCOrder(AxesShape, FixedIdx)
  V = Values(Flat)
  RETURN
END IF

! Grid1D is ALLOCATABLE (rather than an automatic array sized from
! Axes(Depth)%Values) specifically so its bound expression is never
! evaluated until after the Depth > NAxes guard above has already
! returned -- an automatic array's size is computed unconditionally at
! procedure entry in Fortran, which would dereference Axes(Depth) out of
! bounds at the leaf level (Depth == NAxes+1) if declared that way.
Grid1D = Axes(Depth)%Values
Qv = MIN(MAX(Query(Depth), Grid1D(1)), Grid1D(SIZE(Grid1D)))

IF (SIZE(Grid1D) == 1) THEN
  V = MultilinearClipped(Axes, AxesShape, Values, Query, Depth + 1, [FixedIdx, 1])
  RETURN
END IF

CALL LocalWindow(Grid1D, Qv, "linear", Lo, Hi)
VLo = MultilinearClipped(Axes, AxesShape, Values, Query, Depth + 1, [FixedIdx, Lo])
VHi = MultilinearClipped(Axes, AxesShape, Values, Query, Depth + 1, [FixedIdx, Hi])
T = (Qv - Grid1D(Lo)) / (Grid1D(Hi) - Grid1D(Lo))
V = VLo + T * (VHi - VLo)
END FUNCTION MultilinearClipped

!==================================================================================================================================
!> Flat (1-based) offset into a C-order (last axis fastest, matching
!> NumPy's default and how MOD_CoverageKernel_IO/export_hdf5.py lay out
!> every grid-shaped array) flattened array, given AxesShape(1:NAxes) and
!> 1-based MultiIdx(1:NAxes).
!==================================================================================================================================
PURE FUNCTION FlatIndexCOrder(AxesShape, MultiIdx) RESULT(Flat)
! INPUT / OUTPUT VARIABLES
INTEGER, INTENT(IN) :: AxesShape(:)
INTEGER, INTENT(IN) :: MultiIdx(:)
INTEGER :: Flat
! LOCAL VARIABLES
INTEGER :: d, Stride
!----------------------------------------------------------------------------------------------------------------------------
Flat = 0
Stride = 1
DO d = SIZE(AxesShape), 1, -1
  Flat = Flat + (MultiIdx(d) - 1) * Stride
  Stride = Stride * AxesShape(d)
END DO
Flat = Flat + 1
END FUNCTION FlatIndexCOrder

END MODULE MOD_CoverageKernel_Interp1D
