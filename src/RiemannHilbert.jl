module RiemannHilbert
    using Base, ApproxFun

export CauchyOperator, cauchy, hilbert, hilbertinverse, cauchyintegral
import ApproxFun
import ApproxFun: PeriodicDomain, BandedShiftOperator, bandinds, dirichlettransform, idirichlettransform!, Curve,CurveSpace,transform,SpaceOperator, rangespace, domainspace


function cauchy(s::Integer,f,z)
    @assert abs(s) == 1
    
    cauchy(s==1,f,z)
end

cauchy(f,z::Vector)=Complex{Float64}[cauchy(f,zk) for zk in z]
cauchy(s::Integer,f,z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]
cauchy(s,f,z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]


include("circlecauchy.jl")
include("periodiclinecauchy.jl")
include("intervalcauchy.jl")
include("singfuncauchy.jl")

end #module


