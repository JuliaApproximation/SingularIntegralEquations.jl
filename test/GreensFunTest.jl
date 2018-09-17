using ApproxFun, SingularIntegralEquations, Test

@testset "GreensFun" begin
    cts = 100
    Γ,ψ = ones(cts),ones(cts)
    [(Γ[k],ψ[k]) = (k*Γ[k-1],ψ[k-1]+1/k) for k=2:cts]

    function FK1(z)
        if abs(z) <=.25
            upd = 1.
            ret = upd
            l=1
            zt2 = (.5z)^2
            while norm(upd) >= eps()
                upd = zt2^l/(Γ[l]*Γ[l+1])
                ret += upd
                l+=1
            end
        else
            ret = 2besseli(1,abs(z))/abs(z)
        end
        .5ret
    end
    function GK1(z)
        if abs(z) <= .25
            upd = 1.
            ret = upd
            l=1
            zt2 = (.5z)^2
            while norm(upd) >= eps()
                upd = zt2^l/(Γ[l]*Γ[l+1])*(ψ[l]+ψ[l+1])
                ret += upd
                l+=1
            end
        else
            abz = abs(z)
            ret = 4/abz^2 - 4(besselk(1,abz)-besseli(1,abz)*(log(abz/2)+γ))/abz
        end
        .5ret
    end

    d = Segment(-1.,1.)
    sp = Space(d)
    wsp = JacobiWeight(-0.5,-0.5,sp)
    ⨍ = DefiniteLineIntegral(wsp)
    x = Fun(identity,d)

    β = -2.0
    G = exp(β*x)

    Kf = (x,y)->(1/π*exp(β/2*(y-x))*β^2/2^2*(FK1(β*(y-x)/2)*(log(abs(β)/4) + γ) - GK1(β*(y-x)/2)/2))/sqrt(1-y^2)
    K = GreensFun(Kf,sp⊗wsp)
    @test K(0.1,0.2) ≈ Kf(0.1,0.2)

    K0f = (x,y)-> x == y ? abs(β)/(4sqrt(1-y^2)) : exp(β/2*(y-x))*abs(β)/2*besseli(1,abs(β*(y-x)/2))/abs(y-x)/sqrt(1-y^2)
    K0 = GreensFun(K0f,CauchyWeight(sp⊗wsp,0))
    @test K0(0.1,0.2) ≈ K0f(0.1,0.2)*logabs(0.1-0.2)/π


    K2f = (x,y)->exp(β/2*(y-x))/sqrt(1-y^2)
    K2 = GreensFun(K2f,CauchyWeight(sp⊗wsp,2))
    @test K2(0.1,0.2) ≈ K2f(0.1,0.2)/(0.1-0.2)^2/π
end
