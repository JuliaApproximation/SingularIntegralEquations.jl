using ApproxFun,SingularIntegralEquations,Base.Test
    import SingularIntegralEquations:stieltjesmoment,stieltjesjacobimoment


c = [0.9731840665678853,0.11644664868790366,0.8961305368364185,0.30663942299763747,0.4564158422422153,0.7262396331533589,0.6448725310785985,0.05623500638232981,0.8452582204677404,0.25385833878392283]

for z in [1.+1.0im,0.1+0.1im]
    o=Fun(one)
    x=Fun(identity)

    @test_approx_eq stieltjes(o)(z) sum(1/(z-x))
    @test_approx_eq stieltjes(x)(z) sum(x/(z-x))

    for w in (sqrt(1-x),sqrt(1+x))
        @test_approx_eq stieltjes(w*o)(z) sum(w/(z-x))
        @test_approx_eq stieltjes(w*x)(z) sum(w*x/(z-x))
    end

    for a in -0.5:0.5:0.5, b in -0.5:0.5:0.5
        f = Fun(WeightedJacobi(a,b),c)
        @test_approx_eq stieltjes(f)(z) sum(f/(z-x))
    end

    f = Fun(WeightedJacobi(0.123,0.456),c)
    @test_approx_eq stieltjes(f)(z) sum(f/(z-x))

    f = Fun(identity,[-1.,0.,1.])
    @test_approx_eq cauchy(sqrt(Fun(one,space(f))-f^2))(z) cauchy(sqrt(1-Fun()^2),z)
end




x=Fun(identity,[im,0.,1.])
w=2/(sqrt(1-x)*sqrt(1+im*x))

for x in (0.9im,0.4im,0.4,0.9)
    #@test_approx_eq cauchy(w,x,true)-cauchy(w,x,false) w(x)
end
