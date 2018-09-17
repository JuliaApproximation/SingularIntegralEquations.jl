using ApproxFun, SingularIntegralEquations, Test
    import SingularIntegralEquations: ⁻, ⁺

@testset "Wiener–Hopf" begin
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


    H = Hilbert(space(G),1)
    @test domainspace(H) == rangespace(H) == space(G)
    C  = Cauchy(-1)


    A=(I+(I-G)*C)

    F=(G-I)[:,1]
    F̃ = vec(F)

    C2=ApproxFun.promotedomainspace(C.ops[2],space(F))

    ApproxFun.testblockbandedoperator(Hilbert(space(F)[1]))
    ApproxFun.testblockbandedoperator(Hilbert(space(F)))
    ApproxFun.testblockbandedoperator(C2)


    @test norm((C*F - [C*F[1];C*F[2]]).coefficients) == 0
    @test norm((C*G - [C*G[1] C*G[3];C*G[2] C*G[4]]).coefficients) == 0
    @test cauchy(F,exp(0.1im)⁻) ≈ (C*F)(exp(0.1im))
    @test cauchy(G,exp(0.1im)⁻) ≈ (C*G)(exp(0.1im))


    V1 = A\F
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
    @test Fun(V1) == V1

    Ṽ = QR\(G-I)
    V  = (I+(I-G)*C)\(G-I)


    @test map(f->f(exp(0.1im)),Array(G-I)) ≈ (G-I)(exp(0.1im))

    @test (G-I)[:,1]==Fun((G-I)[:,1],rangespace(QR))
    @test (G-I)[:,1]==Fun(vec((G-I)[:,1]),rangespace(QR))

    @test norm((V-Ṽ).coefficients) == 0

    @test norm((V1-V[:,1]).coefficients) == 0

    V2  = A\(G-I)[:,2]
    @test norm((V2-V[:,2]).coefficients) == 0

    @test norm((A*V[:,1]-(G[:,1]-[1,0])).coefficients) < 100eps()

    z=exp(0.1im)
    @test V(z)+(I-G(z))*cauchy(V,(z)⁻) ≈ G(z)-I

    @test cauchy(V[1,1],exp(0.1im)⁻) ≈ (C*V[1,1])(exp(0.1im))
    @test cauchy(V[2,1],exp(0.1im)⁻) ≈ (C*V[2,1])(exp(0.1im))
    @test cauchy(V[:,1],exp(0.1im)⁻) ≈ (C*V[:,1])(exp(0.1im))
    @test cauchy(V,exp(0.1im)⁻) ≈ (C*V)(exp(0.1im))

    Φmi = I+C*V
    Φp = V+Φmi


    @test Φmi(z) ≈ (I+cauchy(V,(z)⁻))
    @test Φp(z) ≈ (I+cauchy(V,(z)⁺))
    @test Φp(z)*inv(Φmi(z)) ≈ G(z)

    Φm=inv.(Φmi)
    @test Φm(z) ≈ inv(Φmi(z))

    T=ToeplitzOperator(G)

    L  = ToeplitzOperator(Φm)
    U  = ToeplitzOperator(Φp)

    @test norm((T-U*L)[1:10,1:10]) < 100eps()  # check the accuracy
end
