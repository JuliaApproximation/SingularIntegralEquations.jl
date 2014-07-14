module RiemannHilbert
    using Base, ApproxFun

export CauchyOperator, cauchy, hilbert, hilbertinverse, cauchyintegral
import ApproxFun
import ApproxFun.PeriodicDomain
import ApproxFun.BandedShiftOperator
import ApproxFun.bandrange
import ApproxFun.dirichlettransform
import ApproxFun.idirichlettransform!


function cauchy(s::Integer,f,z)
    @assert abs(s) == 1
    
    cauchy(s==1,f,z)
end

include("circlecauchy.jl")
include("periodiclinecauchy.jl")
include("singfuncauchy.jl")

end #module


