
__precompile__()
module SingularIntegralEquations
    using Base, ApproxFun, Compat

export cauchy, cauchyintegral, stieltjes, logkernel,
       stieltjesintegral, hilbert, pseudohilbert, pseudocauchy


import Base: values,getindex,setindex!,*,.*,+,.+,-,.-,==,<,<=,>,
                >=,./,/,.^,^,\,∪,transpose


import ApproxFun
import ApproxFun: bandinds,SpaceOperator,
                  plan_transform,plan_itransform,transform,itransform,transform!,itransform!,
                  rangespace, domainspace, addentries!, BandedOperator, AnySpace,
                  canonicalspace, domain, promotedomainspace, promoterangespace, AnyDomain, CalculusOperator,
                  SumSpace,PiecewiseSpace, interlace,Multiplication,ArraySpace,DiagonalArrayOperator,
                  BandedMatrix,bazeros,ChebyshevDirichlet,PolynomialSpace,AbstractProductSpace,evaluate,order,
                  RealBasis,ComplexBasis,AnyBasis,UnsetSpace,ReImSpace,ReImOperator,BivariateFun,linesum,complexlength,
                  ProductFun, LowRankFun, mappoint, PeriodicLineSpace, PeriodicLineDirichlet,Recurrence, FiniteFunctional,
                  real, UnivariateSpace, setdomain, eps, choosedomainspace, isapproxinteger, BlockOperator,
                  ConstantSpace,ReOperator,DirectSumSpace,TupleSpace, AlmostBandedOperator, ZeroSpace,
                  DiagonalInterlaceOperator, LowRankPertOperator

function cauchy(s,f,z)
    if isa(s,Bool)
        error("Override cauchy for "*string(typeof(f)))
    end

    @assert abs(s) == 1

    cauchy(s==1,f,z)
end

hilbert(f)=Hilbert()*f
hilbert(f,z)=hilbert(f)(z)

#TODO: cauchy ->stieljtjes
#TODO: stieltjes -> offhilbert
stieltjes(s,f,z)=-2π*im*cauchy(s,f,z)
stieltjes(f,z)=-2π*im*cauchy(f,z)
cauchyintegral(u,z)=im/(2π)*stieltjesintegral(u,z)


include("LinearAlgebra/LinearAlgebra.jl")
include("Operators/Operators.jl")

include("stieltjesmoment.jl")

include("circlecauchy.jl")
include("intervalcauchy.jl")
include("singfuncauchy.jl")

include("vectorcauchy.jl")

include("GreensFun/GreensFun.jl")

include("periodicline.jl")
include("arc.jl")
include("curve.jl")
include("asymptotics.jl")

include("fractals.jl")

if isdir(Pkg.dir("TikzGraphs"))
    include("introspect.jl")
end


end #module
