#
# Below:
#
# x = trg is the target variable,
# y = src is the source variable,
# E is the energy, and
# derivs allows for calculation of partial derivatives as well.
#

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
