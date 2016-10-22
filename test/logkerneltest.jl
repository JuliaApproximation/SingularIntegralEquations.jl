using Base.Test, ApproxFun, SingularIntegralEquations
    import ApproxFun: ∞



## LogKernel for Legendre and Jacobi on intervel

x=Fun()
f=exp(x)
@test isa(logkernel(f,2.0+im),Real)
@test_approx_eq logkernel(f,2.0+im) sum(f*log(abs(2.0+im-x)))/π


f=sqrt(1-x^2)*exp(x)
@test_approx_eq logkernel(f,2.0+im) sum(f*log(abs(2.0+im-x)))/π

x=Fun()
f=(1-x)^0.1
sp=space(f)
@test_approx_eq logjacobimoment(sp.β,sp.α,2.0) sum((1-x)^sp.β*(1+x)^sp.α*log(abs(2.0-x)))

@test_approx_eq logkernel(f,2.0) sum(f*log(abs(2.0-x))/π)


f=(1-x)^0.1*exp(x)

@test_approx_eq stieltjes(f,2.0) sum(f/(2.0-x))

@test_approx_eq logkernel(f,2.0) sum(f*log(abs(2.0-x))/π)

@test isa(logkernel(f,2.0+im),Real)
@test_approx_eq logkernel(f,2.0+im) sum(f*log(abs(2.0+im-x))/π)

f=(1-x^2)^0.1*exp(x)
sp=space(f)
@test_approx_eq logjacobimoment(sp.β,sp.α,2.0)  sum((1-x^2)^0.1*log(abs(2.0-x)))
@test_approx_eq logkernel(f,2.0+im)  sum(f*log(abs(2.0+im-x))/π)


f=(1-x)^(-0.1)*(1+x)^(-0.2)*exp(x)
@test_approx_eq logkernel(f,2.0+im)  sum(f*log(abs(2.0+im-x))/π)
@test isa(logkernel(f,2.0+im),Real)
