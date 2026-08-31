!==================================================================================================================================
!> GENERIC: Cox-de Boor B-spline basis evaluation. Nothing here knows about
!> coverage or any other domain concept -- pure basis-function math, reused
!> by MOD_CoverageKernel_Classifier for every spline-expanded term.
!>
!> Mirrors bspline.py exactly: same knot-vector convention, same
!> "extrapolation freezes at the boundary span" rule, same "drop the LAST
!> raw basis function" rule (sklearn's SplineTransformer(include_bias=
!> FALSE) convention, verified against sklearn directly in the Python
!> reference implementation -- NOT the first, as might be guessed).
!>
!> Implements the standard Cox-de Boor recursion (The NURBS Book, A2.2
!> BasisFuns) so evaluating one point only touches the `degree+1` basis
!> functions that are nonzero there, never the full knot vector.
!==================================================================================================================================
MODULE MOD_CoverageKernel_BSpline
! MODULES
USE MOD_CoverageKernel_Types, ONLY: wp
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
PUBLIC :: UniformKnots
PUBLIC :: NBasisRaw
PUBLIC :: NonzeroBasis
PUBLIC :: BasisVector
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> Build the extended knot vector sklearn's SplineTransformer(knots=
!> "uniform") produces for NKnots interior knots evenly spaced over
!> [XMin, XMax], extended by Degree extra knots on each side at the same
!> spacing.
!==================================================================================================================================
FUNCTION UniformKnots(XMin, XMax, NKnots, Degree) RESULT(Knots)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: XMin, XMax
INTEGER,  INTENT(IN) :: NKnots, Degree
REAL(wp), ALLOCATABLE :: Knots(:)
! LOCAL VARIABLES
REAL(wp) :: Spacing
REAL(wp) :: Interior(NKnots)
INTEGER  :: i
!----------------------------------------------------------------------------------------------------------------------------
DO i = 1, NKnots
  Interior(i) = XMin + (XMax - XMin) * REAL(i - 1, wp) / REAL(NKnots - 1, wp)
END DO
Spacing = Interior(2) - Interior(1)

ALLOCATE(Knots(NKnots + 2 * Degree))
DO i = 1, Degree
  Knots(i) = Interior(1) - Spacing * REAL(Degree - i + 1, wp)
END DO
Knots(Degree + 1 : Degree + NKnots) = Interior
DO i = 1, Degree
  Knots(Degree + NKnots + i) = Interior(NKnots) + Spacing * REAL(i, wp)
END DO
END FUNCTION UniformKnots

!==================================================================================================================================
!> Number of raw B-spline basis functions for this knot vector (before
!> dropping one for the include_bias=FALSE convention).
!==================================================================================================================================
PURE FUNCTION NBasisRaw(Knots, Degree) RESULT(N)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: Knots(:)
INTEGER,  INTENT(IN) :: Degree
INTEGER :: N
!----------------------------------------------------------------------------------------------------------------------------
N = SIZE(Knots) - Degree - 1
END FUNCTION NBasisRaw

!==================================================================================================================================
!> Evaluate the Degree+1 raw basis functions nonzero at X. Values(j) is raw
!> basis function index (Span - Degree + j), j = 0..Degree (0-based
!> reasoning; returned Values is 1-based, Values(j+1)). Points outside
!> [Knots(Degree+1), Knots(NTotal+1)] (1-based) are clamped to the nearest
!> boundary span and frozen there, matching extrapolation="constant".
!==================================================================================================================================
SUBROUTINE NonzeroBasis(Knots, Degree, XIn, Span, Values)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN)  :: Knots(:)
INTEGER,  INTENT(IN)  :: Degree
REAL(wp), INTENT(IN)  :: XIn
INTEGER,  INTENT(OUT) :: Span                !> 1-based span index (Fortran convention throughout)
REAL(wp), INTENT(OUT) :: Values(0:Degree)
! LOCAL VARIABLES
INTEGER  :: NTotal, j, r
REAL(wp) :: X, LoX, HiX, Saved, Temp
REAL(wp) :: Left(0:Degree), Right(0:Degree)
!----------------------------------------------------------------------------------------------------------------------------
NTotal = NBasisRaw(Knots, Degree)
LoX = Knots(Degree + 1)
HiX = Knots(NTotal + 1)
X = XIn

IF (X <= LoX) THEN
  Span = Degree + 1        ! 1-based: raw basis index (Degree+1) in 1-based numbering == 0-based "Degree"
  X = LoX
ELSE IF (X >= HiX) THEN
  Span = NTotal
  X = HiX
ELSE
  ! 0-based span s such that Knots(s+1) <= X < Knots(s+2) (0-based knots), clamped to [Degree, NTotal-1];
  ! translate directly to the 1-based Span used here.
  Span = FindSpan1Based(Knots, X)
  Span = MAX(Degree + 1, MIN(Span, NTotal))
END IF

Values = 0.0_wp
Values(0) = 1.0_wp
DO j = 1, Degree
  Left(j)  = X - Knots(Span + 1 - j)
  Right(j) = Knots(Span + j) - X
  Saved = 0.0_wp
  DO r = 0, j - 1
    Temp = Values(r) / (Right(r + 1) + Left(j - r))
    Values(r) = Saved + Right(r + 1) * Temp
    Saved = Left(j - r) * Temp
  END DO
  Values(j) = Saved
END DO
END SUBROUTINE NonzeroBasis

!==================================================================================================================================
!> 1-based index i such that Knots(i) <= X < Knots(i+1) (searchsorted-right
!> minus one, translated to 1-based indexing). Internal helper for
!> NonzeroBasis's interior branch only -- boundary clamping is handled by
!> the caller.
!==================================================================================================================================
PURE FUNCTION FindSpan1Based(Knots, X) RESULT(Span)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: Knots(:)
REAL(wp), INTENT(IN) :: X
INTEGER :: Span
! LOCAL VARIABLES
INTEGER :: n, i
!----------------------------------------------------------------------------------------------------------------------------
n = SIZE(Knots)
Span = 1
DO i = 1, n
  IF (Knots(i) <= X) THEN
    Span = i
  ELSE
    EXIT
  END IF
END DO
END FUNCTION FindSpan1Based

!==================================================================================================================================
!> Full basis vector of length NFeaturesOut (== NBasisRaw(Knots,Degree) - 1,
!> the sklearn include_bias=FALSE convention of dropping the LAST raw basis
!> function). This is the feature block fed to the classifier's
!> standardization/coefficient step, so its length/ordering must match what
!> was used at fit time (read from HDF5, not recomputed).
!==================================================================================================================================
FUNCTION BasisVector(Knots, Degree, X, NFeaturesOut) RESULT(Out)
! INPUT / OUTPUT VARIABLES
REAL(wp), INTENT(IN) :: Knots(:)
INTEGER,  INTENT(IN) :: Degree
REAL(wp), INTENT(IN) :: X
INTEGER,  INTENT(IN) :: NFeaturesOut
REAL(wp) :: Out(NFeaturesOut)
! LOCAL VARIABLES
INTEGER  :: Span, j, Idx
REAL(wp) :: Values(0:Degree)
!----------------------------------------------------------------------------------------------------------------------------
CALL NonzeroBasis(Knots, Degree, X, Span, Values)
Out = 0.0_wp
DO j = 0, Degree
  Idx = Span - Degree + j              ! 1-based raw basis function index
  IF (Idx <= NFeaturesOut) THEN        ! drops contributions from the dropped last raw basis function
    Out(Idx) = Values(j)
  END IF
END DO
END FUNCTION BasisVector

END MODULE MOD_CoverageKernel_BSpline
