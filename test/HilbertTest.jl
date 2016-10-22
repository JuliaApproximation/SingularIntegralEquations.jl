using Base.Test, ApproxFun, SingularIntegralEquations
    import ApproxFun: ∞, bandedoperatortest, functionaltest


## Sqrt singularity

S=JacobiWeight(0.5,0.5,Ultraspherical(1,[-2.,-1.]))

@test rangespace(SingularIntegral(S,0))==Chebyshev([-2.,-1.])

f=Fun(x->exp(x)*sqrt(x+2)*sqrt(-1-x),S)

bandedoperatortest(SingularIntegral(S,0))
bandedoperatortest(SingularIntegral(S,1))

@test_approx_eq (SingularIntegral(S,0)*f)(-1.2) logkernel(f,-1.2)
@test_approx_eq (SingularIntegral(S,1)*f)(-1.2) hilbert(f,-1.2)
@test_approx_eq logkernel(f,-1.2) -0.05044654410790341  # Mathematica
@test_approx_eq hilbert(f,-1.2) -0.057515957831535571  # Mathematica

x=Fun(domain(S))
@test_approx_eq sum(f*log(abs(x-2.0)))/π logkernel(f,2.0)

x=Fun(S)

S=JacobiWeight(-0.5,-0.5,Chebyshev([-2.,-1.]))

bandedoperatortest(SingularIntegral(S,0))
bandedoperatortest(SingularIntegral(S,1))
bandedoperatortest(Hilbert(S))


@test rangespace(SingularIntegral(S,0))==Chebyshev([-2.,-1.])

f=Fun(x->exp(x)/(sqrt(x+2)*sqrt(-1-x)),S)

@test_approx_eq (SingularIntegral(S,0)*f)(-1.2) logkernel(f,-1.2)
@test_approx_eq (SingularIntegral(S,1)*f)(-1.2) hilbert(f,-1.2)


@test_approx_eq logkernel(f,-1.2) -0.39563660592242765  # Mathematica
@test_approx_eq hilbert(f,-1.2) 0.26527878405434321204  # Mathematica


x=Fun(identity)
f=(exp(x)/(sqrt(1-x)*sqrt(x+1)))
@test_approx_eq (Hilbert(f|>space,0)*f)(.1) (-0.8545003781055088)
@test_approx_eq (Hilbert(0)*f)(.1) (-0.8545003781055088)
@test_approx_eq (Hilbert()*f)(.1) 1.1404096104609646386

x=Fun(identity,[-1,2])
f=(exp(x)/(sqrt(2-x)*sqrt(x+1)))
@test_approx_eq (Hilbert(f|>space,0)*f)(.1) 0.49127801561694168644
@test_approx_eq (Hilbert(0)*f)(.1) 0.49127801561694168644
@test_approx_eq (Hilbert()*f)(.1) 1.6649936695644078289



x=Fun(identity)
f=(exp(x)*(sqrt(1-x)*sqrt(x+1)))
@test_approx_eq (Hilbert()*f)(.1) 0.43723982258866913063

x=Fun(identity,[-1,2])
f=(exp(x)*(sqrt(2-x)*sqrt(x+1)))
@test_approx_eq (Hilbert()*f)(.1) 2.1380903070701673244

x=Fun(identity)
d=domain(x)
w=1/sqrt(1-x^2)
H=Hilbert(d)
B=ldirichlet(d)

for a in [sqrt(sqrt(5)-2)/2,1.,10.]
    L=H[w]+1/a/sqrt(1+a^2)*x
    u=[B,L]\[1.]
    usol = (1+a^2)/(x^2+a^2)
    @test norm(u-usol) <= eps(1000/a)
end


x = Fun(identity)
w = 1/sqrt(1-x^2)
H = Hilbert(space(w))

bandedoperatortest(H[w])

@test_approx_eq  (H[w]*exp(x))(.1) hilbert(w*exp(x))(.1)


x = Fun(identity)
w = sqrt(1-x^2)
H = Hilbert(space(w))

bandedoperatortest(H[w])

@test_approx_eq (H[w]*exp(x))(.1) hilbert(w*exp(x))(.1)


println("Stieltjes test")

ds1 = JacobiWeight(-.5,-.5,ApproxFun.ChebyshevDirichlet{1,1}())
ds2 = JacobiWeight(-.5,-.5,Chebyshev())
rs = Chebyshev([2.,4.+3im])
f1 = Fun(x->exp(x)/sqrt(1-x^2),ds1)
f2 = Fun(x->exp(x)/sqrt(1-x^2),ds2)
S = Stieltjes(ds1,rs)

bandedoperatortest(S)

z = 3.+1.5im
@test_approx_eq (S*f1)(z) stieltjes(f2,z) #val,err = quadgk(x->f1(x)./(z-x),-1.,1.)
# Operator 1.1589646343327578 - 0.7273679005911196im
# Function 1.1589646343327455 - 0.7273679005911283im


ds1 = JacobiWeight(.5,.5,Ultraspherical(1))
ds2 = JacobiWeight(.5,.5,Chebyshev())
rs = Chebyshev([2.,4.])
f1 = Fun(x->exp(x)*sqrt(1-x^2),ds1)
f2 = Fun(x->exp(x)*sqrt(1-x^2),ds2)
S = Stieltjes(ds1,rs)

bandedoperatortest(S)

z = 3.
@test_approx_eq (S*f1)(z) stieltjes(f2,z) #val,err = quadgk(x->f1(x)./(z-x),-1.,1.;reltol=eps())
# Operator 0.6616422557285478 + 0.0im
# Function 0.661642255728541 - 0.0im

println("Stieltjes integral test")

ds1 = JacobiWeight(-.5,-.5,ApproxFun.ChebyshevDirichlet{1,1}())
ds2 = JacobiWeight(-.5,-.5,Chebyshev())
rs = Chebyshev([2.,4.])
f1 = Fun(x->exp(x)/sqrt(1-x^2),ds1)
f2 = Fun(x->exp(x)/sqrt(1-x^2),ds2)
S = Stieltjes(ds1,rs,0)

bandedoperatortest(S)

z = 3.
@test_approx_eq (S*f1)(z) SingularIntegralEquations.stieltjesintegral(f2,z)
# Operator 3.6322473044237698 + 0.0im
# Function 3.6322473044237515

ds1 = JacobiWeight(.5,.5,Ultraspherical(1))
ds2 = JacobiWeight(.5,.5,Chebyshev())
rs = Chebyshev([2.0,4.0])
f1 = Fun(x->exp(x)*sqrt(1-x^2),ds1)
f2 = Fun(x->exp(x)*sqrt(1-x^2),ds2)
S = Stieltjes(ds1,rs,0)

bandedoperatortest(S)

z = 3.0
@test_approx_eq (S*f1)(z) SingularIntegralEquations.stieltjesintegral(f2,z)
# Operator 1.7772163062194861 + 0.0im
# Function 1.7772163062194637

println("Cauchy test")


f2=Fun(sech,PeriodicLine())
@test_approx_eq cauchy(f2,1.+im) (0.23294739894134472 + 0.10998776661109881im )
@test_approx_eq cauchy(f2,1.-im) (-0.23294739894134472 + 0.10998776661109881im )


f=Fun(sech,Line())
@test_approx_eq cauchy(f,1.+im) cauchy(f2,1.+im)


f=Fun(z->exp(exp(0.1im)*z+1/z),Laurent(Circle()))

@test_approx_eq hilbert(f,exp(0.2im)) hilbert(Fun(f,Fourier),exp(0.2im))
@test_approx_eq hilbert(f,exp(0.2im)) -hilbert(reverseorientation(f),exp(0.2im))
@test_approx_eq hilbert(f,exp(0.2im)) -hilbert(reverseorientation(Fun(f,Fourier)),exp(0.2im))


@test_approx_eq cauchy(f,0.5exp(0.2im)) cauchy(Fun(f,Fourier),0.5exp(0.2im))
@test_approx_eq cauchy(f,0.5exp(0.2im)) -cauchy(reverseorientation(f),0.5exp(0.2im))
@test_approx_eq cauchy(f,0.5exp(0.2im)) -cauchy(reverseorientation(Fun(f,Fourier)),0.5exp(0.2im))
@test_approx_eq cauchy(f,0.5exp(0.2im)) (OffHilbert(space(f),Laurent(Circle(0.5)))*f)(0.5exp(0.2im))/(2im)

bandedoperatortest(OffHilbert(space(f),Laurent(Circle(0.5))))

f=Fun(z->exp(exp(0.1im)*z+1/(z-1.)),Laurent(Circle(1.,0.5)))
@test_approx_eq cauchy(f,0.5exp(0.2im)) cauchy(Fun(f,Fourier),0.5exp(0.2im))
@test_approx_eq cauchy(f,0.5exp(0.2im)) -cauchy(reverseorientation(f),0.5exp(0.2im))
@test_approx_eq cauchy(f,0.5exp(0.2im)) -cauchy(reverseorientation(Fun(f,Fourier)),0.5exp(0.2im))


Γ=Circle()∪Circle(0.5)
f=depiece([Fun(z->z^(-1),Γ[1]),Fun(z->z,Γ[2])])
A=I-(f-Fun(one,space(f)))*Cauchy(-1)
u=A\(f-Fun(one,space(f)))

@test_approx_eq 1+cauchy(u,.1) 1
@test_approx_eq 1+cauchy(u,.8) 1/0.8
@test_approx_eq 1+cauchy(u,2.) 1

c1=0.5+0.1;r1=3.;
c2=-0.1+.2im;r2=0.3;
d1=Circle(c1,r1)
d2=Circle(c2,r2)
z=Fun(identity,d2);
C=Cauchy(Space(d1),Space(d2))

bandedoperatortest(C)

@test norm((C*Fun(exp,d1)-Fun(exp,d2)).coefficients)<100eps()

C2=Cauchy(Space(d2),Space(d1))
bandedoperatortest(C2)

@test norm((C2*Fun(z->exp(1/z)-1,d2)+Fun(z->exp(1/z)-1,d1)).coefficients)<100000eps()

c1=0.1+.1im;r1=.4;
c2=-2.+.2im;r2=0.3;
d1=Circle(c1,r1)
d2=Circle(c2,r2)
@test norm((Cauchy(d1,d2)*Fun(z->exp(1/z)-1,d1)+Fun(z->exp(1/z)-1,d2)).coefficients)<2000eps()

bandedoperatortest(Cauchy(d1,d2))

# complex contour


#Legendre uses FastGaussQuadrature
f=Fun(exp,Legendre())
#@test_approx_eq cauchy(f,.1+0.000000000001im) cauchy(f,.1,+)
#@test_approx_eq cauchy(f,.1-0.000000000001im) cauchy(f,.1,-)
#@test_approx_eq (cauchy(f,.1,+)-cauchy(f,.1,-)) exp(.1)

ω=2.
d=Interval(0.5im,30.0im/ω)
x=Fun(identity,Legendre(d))
@test_approx_eq cauchy(exp(im*ω*x),1.+im) (-0.025430235512791915911 + 0.0016246822285867573678im)


println("Arc test")

a=Arc(0.,1.,0.,π/2)
ζ=Fun(identity,a)
f=Fun(exp,a)*sqrt(abs((ζ-1)*(ζ-im)))
z=.1+.2im
#@test_approx_eq cauchy(f,z) sum(f/(ζ-z))/(2π*im)
z=exp(.1im)
@test_approx_eq hilbert(f,z) im*(cauchy(f,z,+)+cauchy(f,z,-))




println("Functional test")
z=.1+.2im
x=Fun(identity)
f=exp(x)*sqrt(1-x^2)

functionaltest(Stieltjes(space(f),z))

@test_approx_eq Stieltjes(space(f),z)*f stieltjes(f,z)

a=Arc(0.,1.,0.,π/2)
ζ=Fun(identity,a)
f=Fun(exp,a)*sqrt(abs((ζ-1)*(ζ-im)))
H=Hilbert()
z=exp(.1im)
@test_approx_eq (H*f)(z) hilbert(f,z)

bandedoperatortest(Hilbert(space(f)))
