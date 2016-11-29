using ApproxFun, SingularIntegralEquations, Base.Test
    import ApproxFun: ∞, testbandedoperator, testfunctional
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

testbandedoperator(SingularIntegral(space(w),0))
testbandedoperator(Hilbert(space(w)))

@test_approx_eq (SingularIntegral(0)*w)(fromcanonical(d,0.1)) logkernel(w,fromcanonical(d,0.1))
@test_approx_eq (Hilbert()*w)(fromcanonical(d,0.1)) hilbert(w,fromcanonical(d,0.1))


f=real(exp(x))
@test_approx_eq logkernel(f,2) linesum(f*log(abs(x-2.)))/π


w=abs(first(d)-x)^0.1*abs(last(d)-x)^0.2
@test_approx_eq logkernel(f*w,2+2im) linesum(f*w*log(abs(2+2im-x)))/π


## cubic

a=1+10*im;b=2-6*im
d=Curve(Fun(x->1+a*x+x^2+b*x^3))


x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))


@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π


w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))

testbandedoperator(SingularIntegral(space(w),0))
testbandedoperator(Hilbert(space(w)))

@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π
@test_approx_eq (SingularIntegral(0)*w)(fromcanonical(d,0.1)) logkernel(w,fromcanonical(d,0.1))
@test_approx_eq (Hilbert()*w)(fromcanonical(d,0.1)) hilbert(w,fromcanonical(d,0.1))



f=Fun(d,[1.,2.,3.])
@test_approx_eq logkernel(f,2) linesum(f*log(abs(x-2.)))/π


w=abs(first(d)-x)^0.1*abs(last(d)-x)^0.2
@test_approx_eq logkernel(f*w,2+2im) linesum(f*w*log(abs(2+2im-x)))/π


## quartic

a=1+10*im;b=2-6*im
d=Curve(Fun(x->1+a*x+x^2+x^3+b*x^4))


x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))

@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π


testbandedoperator(SingularIntegral(space(w),0))
testbandedoperator(Hilbert(space(w)))

w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))

testbandedoperator(SingularIntegral(space(w),0))
testbandedoperator(Hilbert(space(w)))

@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
@test_approx_eq logkernel(w,2.) linesum(w*log(abs(x-2.)))/π
@test_approx_eq (SingularIntegral(0)*w)(fromcanonical(d,0.1)) logkernel(w,fromcanonical(d,0.1))
@test_approx_eq (Hilbert()*w)(fromcanonical(d,0.1)) hilbert(w,fromcanonical(d,0.1))



## Arc

d=exp(im*Segment(0.1,0.2))
x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))

testbandedoperator(SingularIntegral(space(w),0))
testbandedoperator(Hilbert(space(w)))

z=10.;
@test_approx_eq sum(w/(x-z))/(2π*im) cauchy(w,z)
@test_approx_eq sum(w*log(z-x))/(-2π*im) cauchyintegral(w,z)
@test_approx_eq linesum(w*log(abs(z-x)))/π logkernel(w,z)

w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))

testbandedoperator(SingularIntegral(space(w),0))
testbandedoperator(Hilbert(space(w)))

@test_approx_eq sum(w/(x-z))/(2π*im) cauchy(w,z)
@test_approx_eq sum(w*log(z-x))/(-2π*im) cauchyintegral(w,z)
@test_approx_eq linesum(w*log(abs(z-x)))/π logkernel(w,z)
@test_approx_eq (SingularIntegral(0)*w)(fromcanonical(d,0.1)) logkernel(w,fromcanonical(d,0.1))
@test_approx_eq (Hilbert()*w)(fromcanonical(d,0.1)) hilbert(w,fromcanonical(d,0.1))


## Legendre singularities

Γ=Curve(Fun(x->x^2+im*x))
x=Fun(Γ)
f=exp(x)

@test_approx_eq cauchy(f,1.0) sum(f/(x-1.0))/(2π*im)
