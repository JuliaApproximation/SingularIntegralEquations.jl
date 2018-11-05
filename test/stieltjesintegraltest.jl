using ApproxFun, SingularIntegralEquations, Test

@testset "Stieltjes Integral" begin
    a=2.0
    xp=Fun(identity,Segment(a,1))
    xm=Fun(identity,-a .. -1)
    up=2.0im/(sqrt(1+xp)*sqrt(a+xp)*sqrt(xp-1)*sqrt(a-xp))
    um=2.0im/(sqrt(-1-xm)*sqrt(a+xm)*sqrt(1-xm)*sqrt(a-xm))

    @test ≈((stieltjesintegral(up,0.5+0.0001)-stieltjesintegral(up,0.5))/0.0001,stieltjes(up,0.5);atol=1E-3)
    @test ≈((stieltjesintegral(um,0.5+0.0001)-stieltjesintegral(um,0.5))/0.0001,stieltjes(um,0.5);atol=1E-3)



    function ellipticintegral(a)
        xp=Fun(identity,Segment(a,1))
        xm=Fun(identity,-a .. -1)
        up=2.0im/(sqrt(1+xp)*sqrt(a+xp)*sqrt(xp-1)*sqrt(a-xp))
        um=2.0im/(sqrt(-1-xm)*sqrt(a+xm)*sqrt(1-xm)*sqrt(a-xm))
        z->-cauchyintegral(up,z)-cauchyintegral(um,z)
    end


    z=Fun(0..0.5)
    @test ellipticintegral(2.0)(0.5) ≈ sum(1/(sqrt(1-z^2)*sqrt(2.0^2-z^2)))
end
