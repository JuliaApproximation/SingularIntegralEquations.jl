using ApproxFun, SingularIntegralEquations, Base.Test


# Scalar case

G=Fun(z->2+cos(z+1/z),Circle()) # the symbol of the Toeplitz operator
T=ToeplitzOperator(G)

C  = Cauchy(-1)
V  = (I+(1-G)*C)\(G-1)

Φmi = 1+C*V
Φp = V+Φmi

L  = ToeplitzOperator(1/Φmi)
U  = ToeplitzOperator(Φp)

@test norm((T-U*L)[1:10,1:10]) < 10eps()  # check the accuracy


# Matrix case

G=Fun(z->[-1 -3; -3 -1]/z +
         [ 2  2;  1 -3] +
         [ 2 -1;  1  2]*z,Circle())


C  = Cauchy(-1)

A=(I+(I-G)*C)


F̃ = (G-I)[:,1]
F=Fun((G-I)[:,1])

@test Fun(F̃,space(F)) == F

V1  = A\F
Ṽ1 = A\F̃


A1=ApproxFun.choosespaces(A,(G-I)[:,1])
A2=ApproxFun.choosespaces(A,Fun((G-I)[:,1]))

@test A1\Fun((G-I)[:,1])  == V1
@test A1\(G-I)[:,1]  == V1

QR=qrfact(A1)

@test QR\Fun((G-I)[:,1]) == V1
@test QR\(G-I)[:,1] == V1

@test norm((V1-Ṽ1).coefficients) == 0
@test norm((A*V1-F).coefficients) < 100eps()

@test norm((F-Fun((G-I)[:,1])).coefficients) == 0



V  = (I+(I-G)*C)\(G-I)

@test norm((V1-Fun(V[:,1])).coefficients) == 0


Φmi = I+C*V
Φp = V+Φmi


T=ToeplitzOperator(G)

L  = ToeplitzOperator(inv(Φmi))
U  = ToeplitzOperator(Φp)


@test norm((T-U*L)[1:10,1:10]) < 100eps()  # check the accuracy
