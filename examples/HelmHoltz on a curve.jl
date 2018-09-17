using ApproxFun, SingularIntegralEquations , Plots
gr()


k = 50.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui = (x,y) -> exp(im*k*(d[1]*x+d[2]*y))

# The Helmholtz Green's function, split into singular and nonsingular pieces.
g1 = (x,y) -> -besselj0(k*abs(y-x))/2
g2 = (x,y) -> x == y ? -(log(k/2)+γ)/2/π + im/4 : im/4*hankelh1(0,k*abs(y-x)) - g1(x,y).*logabs(y-x)/π



Γ = Curve(Fun(x->x+im*x^3))

#Γ = Segment(0.,1.0+im)

sp = Space(Γ)
cwsp = CauchyWeight(sp⊗sp,0)
uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(Γ)

@time G = GreensFun(g1,cwsp;method=:Cholesky) + GreensFun(g2,sp⊗sp;method=:Cholesky)

@time ∂u∂n = ⨍[G]\uiΓ


N=100
x = linspace(-3,3,N);y = linspace(-2,2,N)

us = (x,y) -> -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))


@time uvals = ui.(x,y') .+ us.(x,y')

contourf(x,y,real(uvals'))
    plot!(Γ)







println("The length of ∂u∂n is: ",ncoefficients(∂u∂n))
us = (x,y) -> -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))

t=0.1;us(t,t^3)-ui(t,t^3)
t=0.2;us(t,t^3)-ui(t,t^3)

t=0.1;us(t,t)+ui(t,t)
t=0.2;us(t,t)+ui(t,t)


Γ = Curve(Fun(x->x+im*x^3)) ∪ Curve(Fun(x->x-1+im*(x^4+4)))

plot(Γ)

sp = Space(Γ)
cwsp = CauchyWeight(sp⊗sp,0)

uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(Γ)

@time G = GreensFun(g1,cwsp;method=:Cholesky) + GreensFun(g2,sp⊗sp;method=:Cholesky)

@time ∂u∂n = ⨍[G]\uiΓ
println("The length of ∂u∂n is: ",ncoefficients(∂u∂n))
us = (x,y) -> -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))



N=300
x = linspace(-4,3,N);y = linspace(-2,6,N)

us = (x,y) -> -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))


@time uvals = ui.(x,y') .+ us.(x,y')


writecsv("/Users/solver/Desktop/uvals.csv",uvals)



contourf(x,y,real(uvals');legend=false,size=(1000,600))
    plot!(Γ;color=:black)
png("scat.png")









3


@time us(0.1,0.2)


writecsv("dudncoeffs.csv",∂u∂n.coefficients)



# on 0.5
dsp = domainspace(DefiniteLineIntegral(Γ))
cfs = eval.(parse.(vec(readcsv("/Users/solver/Desktop/dudncoeffs.csv"))))

∂u∂n = Fun(dsp,chop!(cfs,1E-5))

us = (x,y) -> -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))

us(0.1,0.2)


N=300
x = linspace(-4,3,N);y = linspace(-2,6,N)
xx,yy = x.+0.*y',0.*x.+y'

@time uvals = ui.(xx,yy) .+ us.(xx,yy)



using Plots
pyplot()

plot(Γ;color="black")



umax = maximum(abs,uvals)
contourf(x,y,real(uvals)')#,cmap="cubehelix")
    plot!(Γ;color="black")

xlabel("\$x\$");ylabel("\$y\$");colorbar(ax=gca(),shrink=0.685)#515#2/3)
savefig("ScatteringNeumannPDEplot.png";dpi=600,bbox_inches="tight")




function makegif(x,y,u,L;plotfunction=Main.PyPlot.contourf,seconds=1,cmap="seismic",vert=1)
    tm=string(time_ns())
    dr = pwd()*"/"*tm*"mov"
    mkdir(dr)

    umax = maxabs(u)
    fps = 24
    MLen = seconds*fps

    tt = linspace(-1.,1.,100)

    for k=1:MLen
        t = 2π/ω*(k-1)/fps
        Main.PyPlot.clf()
        Main.PyPlot.axes(aspect="equal")
        Main.PyPlot.plot(tt,tt.^3;color=:black)
        Main.PyPlot.plot(tt-1,tt.^4.+4;color=:black)
        plotfunction(x,y,real(u*exp(-im*ω*t))';vmin=-umax*vert,vmax=umax*vert,cmap="seismic")
        xlabel!("\$x\$");ylabel!("\$y\$")
        Main.PyPlot.savefig(dr * "/" * lpad(k,max(4,ceil(Int,log10(MLen))),0) * ".png";dpi=150,bbox_inches="tight")
    end
    # Requires: brew install imagemagick
    run(`convert -delay 6 -loop 0 $dr/*.png $dr/scattering.gif`)
    run(`open $dr/scattering.gif`)
end

makegif(x,y,uvals,100;seconds=1,cmap="seismic",vert=0.5)




fps = 24
seconds=1
MLen = seconds*fps

k=MLen
u=uvals;
x=xx;y=yy;
k=24
    t = 2π/ω*(k-1)/fps
    real(u*exp(-im*ω*t))
    Main.PyPlot.contourf(x,y,real(u*exp(-im*ω*t))')
    plot!(Γ;color=:black,legend=false)
xlabel!("\$x\$");ylabel!("\$y\$")

tm=string(time_ns())
dr = pwd()*"/"*tm*"mov"
mkdir(dr)

Main.PyPlot.savefig(dr * "/" * lpad(k,max(4,ceil(Int,log10(MLen))),0) * ".png";dpi=150,bbox_inches="tight")
