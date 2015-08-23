using ApproxFun,SingularIntegralEquations
using Gadfly


##
#  Ideal fluid flow consists of level sets of the imagainary part of a function
# that is asymptotic to c*z and whose imaginary part vanishes on Γ
#
#
# On the unit interval, -2*hilbert(ui) gives the imaginary part of cauchy(ui,z)
#  So if we want to find ui defined on Γ so that hilbert(ui) = imag(c*z)
#  then c*z + 2cauchy(u,z) vanishes on Γ
##



u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


k=101;
    Γ=Interval(0.,1+0.5im)
    z=Fun(Γ)
    α=exp(-π*k/50im)
    c,ui=[1 Hilbert()]\imag(α*z)
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))
##
# On an arc, the Hilbert transform no longer gives the imaginary part
#  However, pseudohilbert gives the imaginary part of pseudocauchy
#  So if we want to find ui defined on Γ so that pseudohilbert(ui) = imag(c*z)
#  then c*z + 2pseudocauchy(u,z) vanishes on Γ
##


u(x,y)=α*(x+im*y)+2pseudocauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=217;
    Γ=0.5+exp(im*Interval(0.1,8))
    z=Fun(Γ)
    α=exp(-k/50im)
    c,ui=[1 PseudoHilbert()]\imag(α*z)
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3:.05:3])))


##
#  Circle
##

Γ=Circle()
z=Fun(Fourier(Γ))


u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=107;
    α=exp(-k/50im)
    c,ui=[0 BasisFunctional(1);
          1 real(Hilbert())]\[0.,imag(α*z)]

    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))

##
# On a curve, the Hilbert transform may be complex, so we
# take the real part
##

u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


Γ=Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
    z=Fun(Γ)
    α=im
    c,ui=[1 real(Hilbert())]\imag(α*z)
    Gadfly.plot(ApproxFun.layer(Γ),
                    layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3:.05:3])))




##
#  Two intervals requires explicitely stating the space (for now)
##

Γ=Interval(-1.,-0.5)∪Interval(0.5,1.)
z=Fun(Γ)



ds=PiecewiseSpace([JacobiWeight(0.5,0.5,Ultraspherical{1}(Γ[1])),
                       JacobiWeight(0.5,0.5,Ultraspherical{1}(Γ[2]))])


    A=[ones(Γ[1]) ones(Γ[2]) real(Hilbert(ds))]
    α=exp(1.0*im)
    P=ApproxFun.interlace(A)


P\imag(α*z)







a,b,ui=ApproxFun.PrependColumnsOperator(A)\imag(c*z)

u2(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'
using Gadfly
Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u2(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3.:.05:3.])))


Γ=Interval(-1.,0.)∪Interval(0.5im,1.)
z=Fun(Γ)


u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=20;
    c=exp(-k/50im)
    ui=[BasisFunctional(1);
        BasisFunctional(2);
        real(Hilbert())]\[0.;0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))




Γ=Interval(-im,1.0-im)∪Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
z=Fun(Γ)
u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

k=20;
    c=exp(-π*k/50im)
    ui=[BasisFunctional(1);
        BasisFunctional(2);
        real(Hilbert())]\Any[0.;0.;imag(c*z)]

m=100;x = linspace(-3.,3.,m);y = linspace(-3.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'
    myplot=Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-5.:.07:4.])))

draw(SVG("/Users/solver/Desktop/myplot.svg", 4inch, 3inch), myplot)

Pkg.add("Cairo")



## Channel


# TODO: rename
using ApproxFun,SingularIntegralEquations,SO
    import ApproxFun:RealUnivariateSpace,ComplexBasis,canonicalspace,bandinds,addentries!,conversion_rule,
            spacescompatible,toeplitz_addentries!,domainscompatible,rangespace
immutable FourierDirichlet <: RealUnivariateSpace
    domain::PeriodicDomain
end

FourierDirichlet()=FourierDirichlet(PeriodicInterval())

    spacescompatible(a::FourierDirichlet,b::FourierDirichlet)=domainscompatible(a,b)

    canonicalspace(S::FourierDirichlet)=Fourier(domain(S))

    bandinds(::Conversion{FourierDirichlet,Fourier})=0,2
    function addentries!(C::Conversion{FourierDirichlet,Fourier},A,kr::Range)
        toeplitz_addentries!([],[1.,0.,1.],A,kr)
    end


    conversion_rule(b::FourierDirichlet,a::Fourier)=b





bandinds{L<:PeriodicLine,T}(H::Hilbert{MappedSpace{FourierDirichlet,L,T}})=-1,3
    rangespace{L<:PeriodicLine,T}(H::Hilbert{MappedSpace{FourierDirichlet,L,T}})=MappedSpace(domain(H),Fourier())

    function addentries!{L<:PeriodicLine,T}(H::Hilbert{MappedSpace{FourierDirichlet,L,T}},A,kr::Range)
        if 1 in kr
            A[1,2]+=1
        end
        addentries!(real(Hilbert(FourierDirichlet(Circle()))),A,kr)
        A
    end

S=MappedSpace(PeriodicLine()-2im,FourierDirichlet())

PS=PiecewiseSpace([S,ChebyshevWeight(ChebyshevDirichlet{1,1}(Interval(2.im,.1+2.im)))])
H=Hilbert(PS)
f=Fun(z->imag(z),rangespace(H))
ui=linsolve([BasisFunctional(1);BasisFunctional(2);real(H)],Any[0.,0.,f];maxlength=1000)
bandinds(H)
length(ui)

f.coefficients
u=Fun([0.;ui.coefficients[2:end]],space(ui))
hilbert(ui,.1-2.im)

hilbert(f,.1-2.im)
(real(H)*f)[.1-2.im]

real(H)[1:10,1:10]

ui=linsolve([real(H)],Any[f];maxlength=10000)
ui=linsolve([BasisFunctional(1);real(H)],Any[0.,f];maxlength=10000)
ui=linsolve([BasisFunctional(2);real(H)],Any[0.,f];maxlength=10000)
ui=linsolve([BasisFunctional(1);BasisFunctional(2);real(H)],Any[0.,0.,f];maxlength=10000)

f=Fun([0.,0.,0.,0.,1.],domainspace(H))
q=H*f

(OH*pieces(f)[1])[2.6im]
(H.op.ops[2,1]*pieces(f)[1])[2.6im]
(H.op.ops[2,2]*pieces(f)[2])[2.6im]
hilbert(f,2.6im)
(H*f)[2.6im]
-stieltjes(pieces(f)[1],2.6im)/π
q[2.6im]
q[-2.im+1.]
ui.coefficients

H22=ApproxFun.SliceOperator(real(Hilbert(PS[2])),1,1,1)
real(Hilbert(PS[2]))[1:10,1:10]
H.op.ops[2,1][1:10,1:10]


ui.coefficients

Hilbert(S)[1:10,1:10]

pieces(f)[1][20.]
(imag(z)|>space)[1]
H[1:20,1:20]



OH.data
f=Fun([zeros(4);1.],S)
    (OH*f)[1.]

-stieltjes(f,1.)/π

v1=chop(Fun(x->-stieltjes(f,x)/π,rs).coefficients,1E-14)



rs=rangespace(OH)
-stieltjes(f,1.)/π

A=rand(2,2)+im*rand(2,2)

convert(ApproxFun.BandedMatrix,A)
methods(OffHilbert)

OH[1:10,1:10]

k=1;f=Fun([zeros(k-1);1],S);(Hilbert(S)*f)[2.]

hilbert(Fun(f,MappedSpace(domain(S),ApproxFun.LaurentDirichlet())),2.)|>chopm

hilbert(Fun([1.],Circle()),exp(0.1im))-hilbert(Fun([1.],Circle()),-1.)
[1:10,1:10]
domain(S)|>last


k=3;
[hilbert(Fun(Fun([zeros(k-1);1.],FourierDirichlet(Circle())),Fourier),-1.) for k=1:10]|>chopm
[cauchy(+1,Fun(Fun([zeros(k-1);1.],FourierDirichlet(Circle())),Fourier),-1.) for k=1:10]|>chopm
Hilbert(FourierDirichlet(Circle()))|>bandinds
Hilbert(FourierDirichlet(Circle()))[1:10,1:10]|>chopm

Hilbert(FourierDirichlet(Circle()))[1:10,1:10]|>chopm

Conversion(FourierDirichlet(),Fourier())[1:10,1:10]|>chopm>





Γ=(PeriodicLine()-im)
d=Circle()
S=ApproxFun.CosDirichlet(d)⊕ApproxFun.SinSpace(d)
S2=Fourier(d)
k=3;cauchy(+1,Fun(Fun([zeros(k-1);1.],FourierDirichlet(Circle())),Fourier),-1.)
k=6;cauchy(+1,Fun(Fun([zeros(k-1);1.],FourierDirichlet(Circle())),Fourier),-1.)
k=10;cauchy(+1,Fun(Fun([zeros(k-1);1.],S),S2),-1.)

C=TimesOperator(Conversion(S2,Laurent(d)),Conversion(S,S2))
C=Conversion(S,S2)

Hilbert(rangespace(C))[1:10,1:10]

TimesOperator(Hilbert(rangespace(C)),C)[1:10,1:10]|>chopm

TimesOperator(Hilbert(rangespace(C)),C)[1:10,1:10]|>chopm
Hilbert(rangespace(C))[1:10,1:10]
C[1:10,1:10]|>chopm

S
[1:10,1:10]


S=MappedSpace(Γ,S)

conversion_type(Chebyshev(),Ultraspherical{1}())
all(map((a,b)->!isa(conversion_type(a,b),NoSpace),S.spaces,S2.spaces))
[map(Conversion,S.spaces,S2.spaces)...]
S2.spaces

Conversion(S,MappedSpace(Γ,ApproxFun.Laurent(PeriodicInterval())))

Hilbert(MappedSpace(Γ,ApproxFun.LaurentDirichlet()))[1:10,1:10]
Hilbert(ApproxFun.PeriodicLineDirichlet(Γ))[1:10,1:10]
