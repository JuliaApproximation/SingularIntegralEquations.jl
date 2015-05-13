#
# This function calculates Φ(x,y,E), the fundamental solution of the Helmholtz equation in a linearly stratified medium:
#
# -(Δ + E + x_2)Φ(x,y,E) = δ(x-y)
#
# also known as the gravity Helmholtz equation.
#
# This is a Julia wrapper for the C-library described in:
#
# A. H. Barnett, B. J. Nelson, and J. M. Mahoney, High-order boundary integral equation solution of high frequency wave scattering from obstacles in an unbounded linearly stratified medium, accepted, J. Comput. Phys., 22 pages (2014).
#
#
# Below:
#
# x = trg is the target variable,
# y = src is the source variable,
# E is the energy, and
# derivs allows for calculation of partial derivatives as well.
#

const lhelmfspath = joinpath(Pkg.dir("SIE"), "deps", "liblhelmfs")

export lhelmfs

function lhelmfs(trg::Union(Vector{Float64},Vector{Complex{Float64}}),energies::Vector{Float64};derivs::Bool=false)
    trgn,energiesn = length(trg),length(energies)
    @assert trgn == energiesn
    n = trgn

    meth = 1
    stdquad = 400
    h = 0.25
    gamout = 0
    nquad = zeros(Int64,1)

    x1,x2 = real(trg),imag(trg)
    u = zeros(Complex{Float64},n)
    #if derivs
        ux = zeros(Complex{Float64},n)
        uy = zeros(Complex{Float64},n)
    #end

    ccall((:lhfs,lhelmfspath),Void,(Ptr{Float64},Ptr{Float64},Ptr{Float64},Int64,Int64,Ptr{Complex{Float64}},Ptr{Complex{Float64}},Ptr{Complex{Float64}},Int64,Float64,Int64,Int64,Ptr{Int64}),x1,x2,energies,derivs ? 1 : 0,n,u,ux,uy,stdquad,h,meth,gamout,nquad)

    if derivs
        return u/4π,ux/4π,uy/4π
    else
        return u/4π
    end
end

function lhelmfs(trg::Union(Matrix{Float64},Matrix{Complex{Float64}}),E::Matrix{Float64};derivs::Bool=false)
    sizetrg,sizeE = size(trg),size(E)
    @assert sizetrg == sizeE

    if derivs
        u,ux,uy = lhelmfs(vec(trg),vec(E);derivs=derivs)
        return reshape(u,sizetrg),reshape(ux,sizetrg),reshape(uy,sizetrg)
    else
        u = lhelmfs(vec(trg),vec(E);derivs=derivs)
        return reshape(u,sizetrg)
    end
end

lhelmfs(trg::Union(VecOrMat{Float64},VecOrMat{Complex{Float64}}),E::Float64;derivs::Bool=false) = lhelmfs(trg,fill(E,size(trg));derivs=derivs)

function lhelmfs(trg::Union(Float64,Complex{Float64}),E::Float64;derivs::Bool=false)
    if derivs
        u,ux,uy = lhelmfs([trg],[E];derivs=derivs)
        return u[1],ux[1],uy[1]
    else
        u = lhelmfs([trg],[E];derivs=derivs)
        return u[1]
    end
end

lhelmfs{T1<:Union(Float64,Complex{Float64}),T2<:Union(Float64,Complex{Float64})}(trg::Union(T1,VecOrMat{T1}),src::Union(T2,VecOrMat{T2}),E::Float64;derivs::Bool=false) = lhelmfs(trg-src,E+imag(src);derivs=derivs)


require("DEQuadrature")

function ψ(a::Number,b::Number,s)
  t = exp(s)
  a./t+b.*t-t.^3/12
end

#=
cp = tan((π^2-6π)/(12+2π))/4
cm = tan((π^2-2π)/(4+2π))/2
c_1 = 1/π+1/2
c_2 = π/4-1/2
c_3 = 1/π+1/6
c_4 = π/12-1/2
=#

cp = tan((big(π)^2-6big(π))/(12+2big(π)))/4
cm = tan((big(π)^2-2big(π))/(4+2big(π)))/2
c_1 = 1/big(π)+1/big(2)
c_2 = big(π)/4-1/big(2)
c_3 = 1/big(π)+1/big(6)
c_4 = big(π)/12-1/big(2)


function gcont(sp,sm,α)
  val1 = c_1*atan(2(α-real(sm)+cm))-c_2
  val2 = c_3*atan(-4(α-real(sp)-cp))-c_4
  val1.*val2
end
function gcontp(sp,sm,α)
  val1 = c_1*atan(2(α-real(sm)+cm))-c_2
  val2 = c_3*atan(-4(α-real(sp)-cp))-c_4
  val3 = c_1*2./(1+(2(α-real(sm)+cm)).^2)
  val4 = c_3*(-4)./(1+(-4(α-real(sp)-cp)).^2)
  val1.*val4+val2.*val3
end

γ(sp,sm,α) = complex(α,gcont(sp,sm,α))
γp(sp,sm,α) = complex(1,gcontp(sp,sm,α))

xDE,wDE = Main.DEQuadrature.DENodesAndWeights(BigFloat("1.0"),[BigFloat("0.0")],2^8;ga=BigFloat("1.0"),domain=Main.DEQuadrature.Infinite2)
#xDE,wDE = Main.DEQuadrature.DENodesAndWeights(1.0,[0.0],2^9;ga=1.5,domain=Main.DEQuadrature.Infinite2)

function gravityhelmholtzfs(trg::Number,energy::Number;n=2^6)

  if n != 2^8
    xDE,wDE = Main.DEQuadrature.DENodesAndWeights(BigFloat("1.0"),[BigFloat("0.0")],n;ga=BigFloat("1.0"),domain=Main.DEQuadrature.Infinite2)
  end

  a,b = trg.^2/4,energy
  tp,tm = sqrt(2(b+sqrt(b^2-a))),sqrt(2(b-sqrt(b^2-a)))
  sp,sm = log(tp),log(tm)

  vals = exp(im*ψ(a,b,γ(sp,sm,xDE))).*γp(sp,sm,xDE)

  cutoff = !isinf(vals).*!isnan(vals).*!isinf(wDE).*!isnan(wDE)
  return float64(dot(conj(vals[cutoff]),wDE[cutoff])/4π)

end

gravityhelmholtzfs{T1<:Union(Float64,Complex{Float64}),T2<:Union(Float64,Complex{Float64})}(trg::Union(T1,VecOrMat{T1}),src::Union(T2,VecOrMat{T2}),E::Float64) = gravityhelmholtzfs(trg-src,E+imag(src))
