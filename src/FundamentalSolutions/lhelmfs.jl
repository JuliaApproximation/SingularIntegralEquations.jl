#
# Below:
#
# x = trg is the target variable,
# y = src is the source variable,
# E is the energy, and
# derivs allows for calculation of partial derivatives as well.
#

function lhelmfs(trg::Union{Float64,Complex{Float64}},energies::Float64;derivs::Bool=false)
    stdquad = 400
    h = 0.25
    meth = 1
    x1,x2 = reim(trg)
    lhfs(x1,x2,energies,derivs,stdquad,h,meth)
end

function lhelmfs(trg::Union{Vector{Float64},Vector{Complex{Float64}}},energies::Vector{Float64};derivs::Bool=false)
    n = length(trg)
    @assert n == length(energies)
    stdquad = 400
    h = 0.25
    meth = 1
    x1,x2 = reim(trg)
    u = Vector{Complex{Float64}}(undef,n)
    if derivs
        ux = Vector{Complex{Float64}}(undef,n)
        uy = Vector{Complex{Float64}}(undef,n)
        lhfs!(u,ux,uy,x1,x2,energies,derivs,stdquad,h,meth,n)
        return u,ux,uy
    else
        lhfs!(u,x1,x2,energies,derivs,stdquad,h,meth,n)
        return u
    end
end

function lhelmfs(trg::Union{Matrix{Float64},Matrix{Complex{Float64}}},E::Matrix{Float64};derivs::Bool=false)
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

lhelmfs(trg::Union{VecOrMat{Float64},VecOrMat{Complex{Float64}}},E::Float64;derivs::Bool=false) =
    lhelmfs(trg,fill(E,size(trg));derivs=derivs)

lhelmfs(trg::Union{T1,VecOrMat{T1}},src::Union{T2,VecOrMat{T2}},E::Float64;derivs::Bool=false) where {T1<:Union{Float64,Complex{Float64}},T2<:Union{Float64,Complex{Float64}}} =
    lhelmfs(trg.-src,E.+imag.(src);derivs=derivs)
