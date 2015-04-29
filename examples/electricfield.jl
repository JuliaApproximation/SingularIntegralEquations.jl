using ApproxFun,SIE

z_0 = 2.0
ui(x,y) = logabs(complex(x,y)-z_0)
g1(x,y) = 1/2

function electricfield(N,r)
    # Set the domains.
    cr = exp(im*2π*[0:N-1]/N)
    crl = (1-2im*r)cr
    crr = (1+2im*r)cr
  dom = ∪(Interval,crl[1:2:end],crr[1:2:end]) ∪ ∪(Circle,cr[2:2:end],ones(length(cr[2:2:end]))r)

    sp = Space(dom)
    cwsp = CauchyWeight(sp⊗sp,0)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t))+0im,sp),DefiniteLineIntegral(dom)

    G = GreensFun(g1,cwsp)
    L,f = ⨍[G],uiΓ

    ∂u∂n = Fun(sparse(ApproxFun.interlace(L)[1:3N,1:3N])\pad(f.coefficients,3N),domainspace(L))

    us(x,y) = -logkernel(∂u∂n,complex(x,y))/2π
    ut(x,y) = ui(x,y) + us(x,y)
    h = 1e-5
    abs((ut(h,0.)-ut(-h,0.))/2h)
end


Nr1 = [5:14]
@time Er1 = Float64[electricfield(N,1e-1) for N in Nr1]
Nr2 = [[5:10],[15:5:20],[30:10:100]]
@time Er2 = Float64[electricfield(N,1e-2) for N in Nr2]
Nr3 = [[5:10],[15:5:20],[30:10:90],[100:50:200]]
@time Er3 = Float64[electricfield(N,1e-3) for N in Nr3]
@time Er4 = Float64[electricfield(N,1e-4) for N in Nr3]
