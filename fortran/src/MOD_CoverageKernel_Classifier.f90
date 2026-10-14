!==================================================================================================================================
!> GENERIC: feature assembly from a t_SplineClassifier's Terms(:), plus
!> present-classes-masked softmax evaluation. Mirrors spline_classifier.py:
!> this module has no idea what the raw feature columns *mean* (coverage,
!> angle, velocity, ...) -- it just builds a standardized feature vector
!> from named terms (B-spline expansion or passthrough) and evaluates a
!> multinomial logistic classifier over it.
!>
!> Classes never observed at fit time are entirely absent from Coef/
!> Intercept (see t_SplineClassifier's docstring in MOD_CoverageKernel_Types)
!> -- EvalSplineClassifier reconstructs the full NClasses-length probability
!> vector by softmaxing only over PresentClasses and scattering zeros
!> elsewhere, never by reading Coef as if it were already dense.
!==================================================================================================================================
MODULE MOD_CoverageKernel_Classifier
! MODULES
USE MOD_CoverageKernel_Types,   ONLY: wp, t_Term, t_SplineClassifier
USE MOD_CoverageKernel_BSpline, ONLY: BasisVector
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
PUBLIC :: BuildFeatureVector
PUBLIC :: EvalSplineClassifier
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> Assemble the raw (pre-standardization) feature vector from XRaw(:) (the
!> classifier's own input columns, 1-based) via its Terms(:): a SplineTerm
!> expands one column into a B-spline basis block, a LinearTerm passes its
!> column through unchanged. Mirrors spline_classifier.py's
!> _term_columns_raw for a single row.
!==================================================================================================================================
FUNCTION BuildFeatureVector(Terms, XRaw, NFeatures) RESULT(Z)
! INPUT / OUTPUT VARIABLES
TYPE(t_Term), INTENT(IN) :: Terms(:)
REAL(wp),     INTENT(IN) :: XRaw(:)
INTEGER,      INTENT(IN) :: NFeatures
REAL(wp) :: Z(NFeatures)
! LOCAL VARIABLES
INTEGER :: t, Offset
!----------------------------------------------------------------------------------------------------------------------------
Offset = 0
DO t = 1, SIZE(Terms)
  ASSOCIATE(Term => Terms(t))
    IF (TRIM(Term%Kind) == "spline") THEN
      Z(Offset + 1 : Offset + Term%NFeaturesOut) = &
        BasisVector(Term%Knots, Term%Degree, XRaw(Term%Column), Term%NFeaturesOut)
    ELSE
      Z(Offset + 1) = XRaw(Term%Column)
    END IF
    Offset = Offset + Term%NFeaturesOut
  END ASSOCIATE
END DO
END FUNCTION BuildFeatureVector

!==================================================================================================================================
!> Evaluate P(class=k | XRaw) for k=1..Clf%NClasses. Builds the feature
!> vector via BuildFeatureVector, standardizes with the fitted
!> ScalerMean/ScalerScale, applies a numerically-stable softmax over only
!> the present (observed-at-fit-time) classes, then scatters the result
!> into the full NClasses-length output (classes absent from the fit are
!> exactly 0, not a small nonzero share -- see this module's docstring).
!==================================================================================================================================
FUNCTION EvalSplineClassifier(Clf, XRaw) RESULT(Probs)
! INPUT / OUTPUT VARIABLES
TYPE(t_SplineClassifier), INTENT(IN) :: Clf
REAL(wp),                 INTENT(IN) :: XRaw(:)
REAL(wp) :: Probs(Clf%NClasses)
! LOCAL VARIABLES
REAL(wp) :: Z(Clf%NFeatures), Zs(Clf%NFeatures)
REAL(wp) :: Logits(Clf%NPresentClasses), MaxLogit, ExpSum
INTEGER  :: p
!----------------------------------------------------------------------------------------------------------------------------
Z = BuildFeatureVector(Clf%Terms, XRaw, Clf%NFeatures)
Zs = (Z - Clf%ScalerMean) / Clf%ScalerScale

DO p = 1, Clf%NPresentClasses
  Logits(p) = DOT_PRODUCT(Zs, Clf%Coef(p, :)) + Clf%Intercept(p)
END DO

MaxLogit = MAXVAL(Logits)
Logits = EXP(Logits - MaxLogit)
ExpSum = SUM(Logits)

Probs = 0.0_wp
DO p = 1, Clf%NPresentClasses
  Probs(Clf%PresentClasses(p)) = Logits(p) / ExpSum
END DO
END FUNCTION EvalSplineClassifier

END MODULE MOD_CoverageKernel_Classifier
