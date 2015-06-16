using ApproxFun,SIE,Base.Test
    import SIE.stieltjesmoment

z=1.+1.im
x=Fun()
@test_approx_eq stieltjesmoment(Jacobi(0.,0.),1,z) sum(1/(z-x))
@test_approx_eq stieltjesmoment(Jacobi(0.,0.),2,z) sum(x/(z-x))

@test_approx_eq stieltjesmoment(JacobiWeight(0.,0.,Legendre()),1,z) sum(1/(z-x))
@test_approx_eq stieltjesmoment(JacobiWeight(0.,0.,Legendre()),2,z) sum(x/(z-x))


@test_approx_eq stieltjesmoment(JacobiWeight(0.,0.5,Jacobi(0.5,0.)),1,z) sum(sqrt(1-x)/(z-x))
@test_approx_eq stieltjesmoment(JacobiWeight(0.,0.5,Jacobi(0.5,0.)),2,z) sum(x*sqrt(1-x)/(z-x))


@test_approx_eq stieltjesmoment(JacobiWeight(0.5,0.,Jacobi(0.5,0.)),1,z) sum(sqrt(1+x)/(z-x))
@test_approx_eq stieltjesmoment(JacobiWeight(0.5,0.,Jacobi(0.5,0.)),2,z) sum(x*sqrt(1+x)/(z-x))



@test_approx_eq cauchy(sqrt(1-Fun(identity,[-1.,0.,1.])^2),1.+1.im) cauchy(sqrt(1-Fun()^2),1.+1.im)
