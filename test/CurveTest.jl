using ApproxFun, SingularIntegralEquations, Base.Test

## quadratic

a=1+10*im;b=2-6*im
d=Curve(Fun(x->1+a*x+b*x^2))


x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))


@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)


## Arc

d=exp(im*Interval(0.1,0.2))
x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))


z=10.;
@test_approx_eq sum(w/(x-z))/(2π*im) cauchy(w,z)
@test_approx_eq sum(w*log(z-x))/(-2π*im) cauchyintegral(w,z)

