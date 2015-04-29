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
function lhelmfs(trg::Matrix{Float64},src::Matrix{Float64},E::Float64,derivs::Bool=false)
    trgn,trg2 = size(trg)
    srcn,src2 = size(src)
    n = trgn
    @assert n == srcn
    @assert trg2 == src2 == 2

    meth = 1
    stdquad = 200
    h = 0.35
    gamout = 0
    nquad = zeros(Int64,1)

    x1 = trg[:,1]-src[:,1]
    x2 = trg[:,2]-src[:,2]
    energies = E+src[:,2]
    u = zeros(Complex{Float64},n)
    #if derivs
        ux = zeros(Complex{Float64},n)
        uy = zeros(Complex{Float64},n)
    #end

    ccall((:lhfs,"/Users/Mikael/.julia/v0.3/SIE/src/lhelmfs/liblhelmfs"),Void,(Ptr{Float64},Ptr{Float64},Ptr{Float64},Int64,Int64,Ptr{Complex{Float64}},Ptr{Complex{Float64}},Ptr{Complex{Float64}},Int64,Float64,Int64,Int64,Ptr{Int64}),x1,x2,energies,derivs ? 1 : 0,n,u,ux,uy,stdquad,h,meth,gamout,nquad)

    if derivs
        return u/4π,ux/4π,uy/4π
    else
        return u/4π
    end
end
