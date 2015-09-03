using ApproxFun,SingularIntegralEquations,Base.Test
    import SingularIntegralEquations:stieltjesmoment,stieltjesjacobimoment


x=Fun()


for z in [1.+1.im,0.1+0.1im]
    @test_approx_eq stieltjesmoment(Jacobi(0.,0.),1,z) sum(1/(z-x))
    @test_approx_eq stieltjesmoment(Jacobi(0.,0.),2,z) sum(x/(z-x))

    @test_approx_eq stieltjesmoment(JacobiWeight(0.,0.,Legendre()),1,z) sum(1/(z-x))
    @test_approx_eq stieltjesmoment(JacobiWeight(0.,0.,Legendre()),2,z) sum(x/(z-x))


    @test_approx_eq stieltjesmoment(JacobiWeight(0.,0.5,Jacobi(0.5,0.)),1,z) sum(sqrt(1-x)/(z-x))
    @test_approx_eq stieltjesjacobimoment(0.,0.5,2,z) sum(x*sqrt(1-x)/(z-x))
    @test_approx_eq stieltjesmoment(JacobiWeight(0.,0.5,Jacobi(0.5,0.)),2,z) sum(Fun([0.,1.],JacobiWeight(0.,0.5,Jacobi(0.5,0.)))/(z-x))
    @test_approx_eq stieltjesmoment(JacobiWeight(0.,0.5,Jacobi(0.5,0.)),2,z) stieltjes(Fun([0.,1.],JacobiWeight(0.,0.5,Jacobi(0.5,0.))),z)


    @test_approx_eq stieltjesmoment(JacobiWeight(0.5,0.,Jacobi(0.5,0.)),1,z) sum(sqrt(1+x)/(z-x))
    @test_approx_eq stieltjesmoment(JacobiWeight(0.5,0.,Jacobi(0.5,0.)),2,z) sum(Fun([0.,1.],JacobiWeight(0.5,0.,Jacobi(0.5,0.)))/(z-x))
    @test_approx_eq stieltjesmoment(JacobiWeight(0.5,0.,Jacobi(0.5,0.)),2,z) stieltjes(Fun([0.,1.],JacobiWeight(0.5,0.,Jacobi(0.5,0.))),z)


    @test_approx_eq stieltjesmoment(JacobiWeight(0.,-0.5,Jacobi(-0.5,0.)),1,z) sum(1/(sqrt(1-x)*(z-x)))
    @test_approx_eq stieltjesmoment(JacobiWeight(0.,-0.5,Jacobi(-0.5,0.)),2,z) sum(Fun([0.,1.],JacobiWeight(0.,-0.5,Jacobi(-0.5,0.)))/(z-x))
    @test_approx_eq stieltjesmoment(JacobiWeight(0.,-0.5,Jacobi(-0.5,0.)),2,z) stieltjes(Fun([0.,1.],JacobiWeight(0.,-0.5,Jacobi(-0.5,0.))),z)


    @test_approx_eq stieltjesmoment(JacobiWeight(-0.5,0.,Jacobi(-0.5,0.)),1,z) sum(1/(sqrt(1+x)*(z-x)))
    @test_approx_eq stieltjesmoment(JacobiWeight(-0.5,0.,Jacobi(-0.5,0.)),2,z) sum(Fun([0.,1.],JacobiWeight(-0.5,0.,Jacobi(-0.5,0.)))/(z-x))
    @test_approx_eq stieltjesmoment(JacobiWeight(-0.5,0.,Jacobi(-0.5,0.)),2,z) stieltjes(Fun([0.,1.],JacobiWeight(-0.5,0.,Jacobi(-0.5,0.))),z)

    f = Fun(identity,[-1.,0.,1.])
    @test_approx_eq cauchy(sqrt(Fun(one,space(f))-f^2),z) cauchy(sqrt(1-Fun()^2),z)
end



