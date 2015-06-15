using ApproxFun,SIE,Base.Test
    import SIE.stieltjesmoment

z=1.+1.im
x=Fun()
@test_approx_eq stieltjesmoment(Jacobi(0.,0.),1,z) sum(1/(z-x))
@test_approx_eq stieltjesmoment(Jacobi(0.,0.),2,z) sum(x/(z-x))


@test_approx_eq stieltjesmoment(JacobiWeight(0.,0.,Legendre()),2,z) sum(x/(z-x))

0==0.0

