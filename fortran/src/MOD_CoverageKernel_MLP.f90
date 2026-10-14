!==================================================================================================================================
!> GENERIC: forward pass of a feed-forward network (dense layers with an
!> activation each). Mirrors nn.py's MLP.logits: this module has no idea
!> what the inputs or outputs mean.
!==================================================================================================================================
MODULE MOD_CoverageKernel_MLP
! MODULES
USE MOD_CoverageKernel_Types, ONLY: wp, t_MLP
IMPLICIT NONE
PRIVATE
!----------------------------------------------------------------------------------------------------------------------------
PUBLIC :: EvalMLP
PUBLIC :: IsKnownActivation
!==================================================================================================================================
CONTAINS

!==================================================================================================================================
!> Output of the network Net for the input vector X (already assembled and
!> standardized by the caller). Returns the last layer's output.
!==================================================================================================================================
FUNCTION EvalMLP(Net, X) RESULT(Y)
! INPUT / OUTPUT VARIABLES
TYPE(t_MLP), INTENT(IN) :: Net
REAL(wp),    INTENT(IN) :: X(:)
REAL(wp), ALLOCATABLE   :: Y(:)
! LOCAL VARIABLES
REAL(wp), ALLOCATABLE :: H(:)
INTEGER :: l
!----------------------------------------------------------------------------------------------------------------------------
H = X
DO l = 1, Net%NLayers
  ASSOCIATE(Layer => Net%Layers(l))
    H = MATMUL(Layer%W, H) + Layer%B
    CALL ApplyActivation(Layer%Activation, H)
  END ASSOCIATE
END DO
Y = H
END FUNCTION EvalMLP

!==================================================================================================================================
!> In-place activation. SiLU is written as x / (1 + exp(-x)), exactly as in
!> nn.py (for x below about -709 exp overflows to +Inf and the result is -0).
!==================================================================================================================================
SUBROUTINE ApplyActivation(Name, H)
! INPUT / OUTPUT VARIABLES
CHARACTER(LEN=*), INTENT(IN)    :: Name
REAL(wp),         INTENT(INOUT) :: H(:)
!----------------------------------------------------------------------------------------------------------------------------
SELECT CASE (TRIM(Name))
CASE ("silu")
  H = H / (1.0_wp + EXP(-H))
CASE ("tanh")
  H = TANH(H)
CASE ("relu")
  H = MAX(H, 0.0_wp)
CASE ("linear")
  CONTINUE
CASE DEFAULT
  ERROR STOP "MOD_CoverageKernel_MLP: unknown activation (expected silu, tanh, relu or linear)"
END SELECT
END SUBROUTINE ApplyActivation

!==================================================================================================================================
!> True if Name is an activation this module can evaluate (checked at load time).
!==================================================================================================================================
PURE FUNCTION IsKnownActivation(Name) RESULT(Known)
CHARACTER(LEN=*), INTENT(IN) :: Name
LOGICAL :: Known
Known = ANY(TRIM(Name) == ["silu  ", "tanh  ", "relu  ", "linear"])
END FUNCTION IsKnownActivation

END MODULE MOD_CoverageKernel_MLP
