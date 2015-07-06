# This file calculates the UL Decomposition of a Toeplitz operator,
# via the Wiener–Hopf factorization.  If the symbol of
# the Toeplitz operator is G(z), we find a Wiener–Hopf factorization
#
# 	G(z) = φ_+(z)*φ_-(z)
#
# where φ_± are analytic in the interior/exterior of the unit
# circle.  Then we have the UL decomposition
#
#   T[G]=T[φ_+]*T[φ_-]
#
# The Wiener–Hopf factorization is found by rephrasing as a
# Riemann–Hilbert problem:
#
#   φ_+(z)=φ_-(z)^(-1)*G(z)
#
# and writing
#
#   φ_-(z)^(-1) = 1 + C_-V(z) and φ_+(z)=1+C_+V(z)=1+V+C_-V(z)
#
# where C_± are the left/right limits of the Cauchy operator.
# This leads to a singular integral equation
#
#   V + C_-V*(1-G) = G - 1
#
# This approach also extends to the matrix case.



using ApproxFun, SingularIntegralEquations


# Scalar case

G=Fun(z->2+cos(z+1/z),Circle()) # the symbol of the Toeplitz operator
T=ToeplitzOperator(G)

C  = Cauchy(-1)
V  = (I+(1-G)*C)\(G-1)

Φmi = 1+C*V
Φp = V+Φmi

L  = ToeplitzOperator(1/Φmi)
U  = ToeplitzOperator(Φp)

norm((T-U*L)[1:10,1:10])  # check the accuracy


# Matrix case

G=Fun(z->[-1 -3; -3 -1]/z +
         [ 2  2;  1 -3] +
         [ 2 -1;  1  2]*z,Circle())
T=ToeplitzOperator(G)

C  = Cauchy(-1)
V  = V=(I+(I-G)*C)\(G-I)

Φmi = I+C*V
Φp = V+Φmi

L  = ToeplitzOperator(inv(Φmi))
U  = ToeplitzOperator(Φp)


norm((T-U*L)[1:10,1:10])  # check the accuracy
