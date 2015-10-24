using ApproxFun, SingularIntegralEquations, Base.Test

## quadratic

a=1+10*im;b=2-6*im
d=Curve(Fun(x->1+a*x+b*x^2))


x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))

@test_approx_eq cauchy(w,2.) (-4.722196879007759+2.347910413861846im)
@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π


w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))

@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π
@test_approx_eq (SingularIntegral(0)*w)(fromcanonical(d,0.1)) logkernel(w,fromcanonical(d,0.1))
@test_approx_eq (Hilbert()*w)(fromcanonical(d,0.1)) hilbert(w,fromcanonical(d,0.1))




## cubic

a=1+10*im;b=2-6*im
d=Curve(Fun(x->1+a*x+x^2+b*x^3))


x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))


@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π


w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))

@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π
@test_approx_eq (SingularIntegral(0)*w)(fromcanonical(d,0.1)) logkernel(w,fromcanonical(d,0.1))
@test_approx_eq (Hilbert()*w)(fromcanonical(d,0.1)) hilbert(w,fromcanonical(d,0.1))



## quartic

a=1+10*im;b=2-6*im
d=Curve(Fun(x->1+a*x+x^2+x^3+b*x^4))


x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))

@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π



w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))

@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π
@test_approx_eq (SingularIntegral(0)*w)(fromcanonical(d,0.1)) logkernel(w,fromcanonical(d,0.1))
@test_approx_eq (Hilbert()*w)(fromcanonical(d,0.1)) hilbert(w,fromcanonical(d,0.1))



## Arc

d=exp(im*Interval(0.1,0.2))
x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))


z=10.;
@test_approx_eq sum(w/(x-z))/(2π*im) cauchy(w,z)
@test_approx_eq sum(w*log(z-x))/(-2π*im) cauchyintegral(w,z)
@test_approx_eq linesum(w*log(abs(z-x)))/π logkernel(w,z)

w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))
@test_approx_eq sum(w/(x-z))/(2π*im) cauchy(w,z)
@test_approx_eq sum(w*log(z-x))/(-2π*im) cauchyintegral(w,z)
@test_approx_eq linesum(w*log(abs(z-x)))/π logkernel(w,z)
@test_approx_eq (SingularIntegral(0)*w)(fromcanonical(d,0.1)) logkernel(w,fromcanonical(d,0.1))
@test_approx_eq (Hilbert()*w)(fromcanonical(d,0.1)) hilbert(w,fromcanonical(d,0.1))
