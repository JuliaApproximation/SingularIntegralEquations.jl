# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0,
# u^i = e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u∂n of the entire wave is calculated on the sound-hard line [-1,1].
# The scattered wave is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE
include("Scatteraux.jl")

k = 50.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))


#=
    dom = Interval(-1.,1.)
    sp = Chebyshev(dom)
    wsp = JacobiWeight(-.5,-.5,sp)
    cwsp = CauchyWeight{0}(sp⊗wsp)
    uiΓ,⨍ = Fun(x->ui(x,0),sp),DefiniteLineIntegral(dom)

    G = ProductFun((x,y)->-besselj0(k*(y-x))/2π,cwsp) + ProductFun((x,y)->(besselj0(k*(y-x))*(im*π/2+log(abs(y-x)))/2π - bessely0(k*abs(y-x))/4)/π,sp,wsp;method=:convolution)
    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))
    us(x,y) = Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂n[t],ApproxFun.ArraySpace(sp,length(x)),length(∂u∂n)).coefficients[1:length(x)]
=#

#=
    N = 6
    r = rand(2N+1)
    cr = cumsum(r)
    ccr = -3+(cr-cr[1])*6/(cr[end]-cr[1]) # For a nice plot, try: [-3.0,-2.4710248798864565,-1.7779535080542614,-0.999257770563108,-0.9160576190726175,-0.5056650643725802,0.7258681480228484,1.2291671942613505,1.3417993440008456,1.485081132919861,1.7601585357456848,2.9542404467603642,3.0]
    dom = ApproxFun.UnionDomain(Interval(ccr+(3-ccr[end-1])/2)[1:2:end])
=#

    N = 5
    dom = Interval(-2.5-.5im,-1.5-.5im)∪Interval(-1.5+.5im,-.5+.5im)∪Interval(-.5-.5im,.5-.5im)∪Interval(.5+.5im,1.5+.5im)∪Interval(1.5-.5im,2.5-.5im)

    sp = Space(dom)
    wsp = ApproxFun.PiecewiseSpace([JacobiWeight(-.5,-.5,sp.spaces[i]) for i=1:N])
    cwsp = [CauchyWeight{0}(sp[i]⊗wsp[i]) for i=1:N]
    xid = Fun(identity,sp)
    uiΓ,⨍ = Fun(t->ui(real(xid[t]),imag(xid[t])),sp),DefiniteLineIntegral(wsp)

    g1(x,y) = -besselj0(k*abs(y-x))/2π
    g2(x,y) = (besselj0(k*abs(y-x))*(im*π/2+log(abs(y-x)))/2π - bessely0(k*abs(y-x))/4)/π
    g3(x,y) = im/4π*hankelh1(0,k*abs(y-x))


    G = Array(GreensFun,N,N)
    for i=1:N,j=1:N
        if i == j
            G[i,i] = ProductFun(g1,cwsp[i]) + ProductFun(g2,sp[i],wsp[i];method=:convolution)
        else
            G[i,j] = GreensFun([ProductFun(g3,sp[i],wsp[j];method=:convolution)])
        end
    end

    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))
    ∂u∂nv = vec(∂u∂n)
function us(x,y)
    ret = Fun(t->-π*g3(x-real(xid[t]),im*(y-imag(xid[t])))*∂u∂nv[1][t],ApproxFun.ArraySpace(sp[1],length(x)),length(∂u∂nv[1])).coefficients[1:length(x)].*length(domain(sp[1]))/2
    for i=2:N
        ret += Fun(t->-π*g3(x-real(xid[t]),im*(y-imag(xid[t])))*∂u∂nv[i][t],ApproxFun.ArraySpace(sp[i],length(x)),length(∂u∂nv[i])).coefficients[1:length(x)]*length(domain(sp[i]))/2
    end
    ret
end
