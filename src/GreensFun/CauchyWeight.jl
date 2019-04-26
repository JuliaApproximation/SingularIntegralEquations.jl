# CauchyWeight enriches a TensorSpace with a singular diagonal

export CauchyWeight

struct CauchyWeight{O,SV,DD,RR} <: AbstractProductSpace{SV,DD,RR}
    space::TensorSpace{SV,DD,RR}
end

CauchyWeight(space::TensorSpace{SV,DD,RR},O) where {SV,DD,RR} = CauchyWeight{O,SV,DD,RR}(space)

order(C::CauchyWeight{O}) where {O} = O
domain(C::CauchyWeight) = domain(C.space)
ncomponents(C::CauchyWeight) = ncomponents(C.space)
component(C::CauchyWeight, k::Integer) = CauchyWeight(component(C.space,k),order(C))
columnspace(C::CauchyWeight, _) = C[1]
component(C::CauchyWeight{O,Tuple{PWS1,PWS2}},i,j) where {O,PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace} =
    CauchyWeight(component(C.space,i,j),O)

factor(s::CauchyWeight,k) = factor(s.space,k) # TODO: Why is this a meaningful definition?
Base.transpose(C::CauchyWeight{O}) where {O} = CauchyWeight(transpose(C.space),O)

cauchyweight(O,x,y) = O == 0 ? logabs(y-x)/π : (y-x)^(-O)/π
cauchyweight(C::CauchyWeight{O},x,y) where {O} = cauchyweight(O,tocanonical(C,x,y)...)






## BivariateFun constructors in a CauchyWeight space

## TODO: for different domains, there should not be (x,y)->f(x,y)*cauchyweight(O,x,y)
## This will change for v0.0.2 with the switch to ChebyshevDirichlet{1,1} bases

for Func in (:ProductFun,:convolutionProductFun)
    @eval begin
        function $Func(f::DFunction,cwsp::CauchyWeight{O};kwds...) where O
            F = domain(factor(cwsp.space,1)) == domain(factor(cwsp.space,2)) ?
                $Func(f,factor(cwsp.space,1),factor(cwsp.space,2);kwds...) :
                $Func((x,y)->f(x,y)*cauchyweight(O,x,y),factor(cwsp.space,1),factor(cwsp.space,2);kwds...)
            return ProductFun(F.coefficients,cwsp)
        end
    end
end

function LowRankFun(f::DFunction,cwsp::CauchyWeight{O};retmax::Bool=false,kwds...) where O
    if retmax
        F,maxabsf = domain(factor(cwsp.space,1)) == domain(factor(cwsp.space,2)) ?
            LowRankFun(f,factor(cwsp.space,1),factor(cwsp.space,2);retmax=retmax,kwds...) :
            LowRankFun((x,y)->f(x,y)*cauchyweight(O,x,y),factor(cwsp.space,1),factor(cwsp.space,2);retmax=retmax,kwds...)
        LowRankFun(F.A,F.B,cwsp),maxabsf
    else
        F = domain(factor(cwsp.space,1)) == domain(factor(cwsp.space,2)) ?
            LowRankFun(f,factor(cwsp.space,1),factor(cwsp.space,2);kwds...) :
            LowRankFun((x,y)->f(x,y)*cauchyweight(O,x,y),factor(cwsp.space,1),factor(cwsp.space,2);kwds...)
        LowRankFun(F.A,F.B,cwsp)
    end
end

## Definite (Line) Integration over BivariateFuns in a CauchyWeight space

## TODO: for different domains, should be OffOp instead of ⨍
## This will change for v0.0.2 with the switch to ChebyshevDirichlet{1,1} bases

for (Func,Op) in ((:DefiniteIntegral,:Hilbert),
                    (:DefiniteLineIntegral,:SingularIntegral))
    @eval begin
        function Base.getindex(⨍::$Func,f::ProductFun{S,V,CauchyWeight{O,Tuple{S,V},T2,DD},T}) where {S,V,O,T,T2,DD}
            if domain(factor(f.space.space,1)) == domain(factor(f.space.space,2))
                $Op(domainspace(⨍),O)[f]
            else
                ⨍[ProductFun(f.coefficients,f.space.space)]
            end
        end
        function Base.getindex(⨍::$Func,f::LowRankFun{S,M,CauchyWeight{O,Tuple{S,M},T2,DD},T}) where {S,M,O,T,T2,DD}
            if domain(factor(f.space.space,1)) == domain(factor(f.space.space,2))
                $Op(domainspace(⨍),O)[f]
            else
                ⨍[LowRankFun(f.A,f.B,f.space.space)]
            end
        end
    end
end




## Evaluation of bivariate functions in a CauchyWeight space

evaluate(f::ProductFun{S,V,CauchyWeight{O,Tuple{S,V},T1,DD},T2},x::AbstractRange,y::AbstractRange) where {S<:UnivariateSpace,V<:UnivariateSpace,O,T1,T2,DD} =
    evaluate(f,[x],[y])
evaluate(f::ProductFun{S,V,CauchyWeight{O,Tuple{S,V},T1,DD},T2},x,y) where {S<:UnivariateSpace,V<:UnivariateSpace,O,T1,T2,DD} =
    evaluate(ProductFun(f.coefficients,space(f).space),x,y).*cauchyweight(space(f),x,y)

+(F::ProductFun{S,V,CauchyWeight{O},T1},G::ProductFun{S,V,CauchyWeight{O},T2}) where {S<:UnivariateSpace,V<:UnivariateSpace,O,T1,T2} =
    ProductFun(ProductFun(F.coefficients,F.space.space)+ProductFun(G.coefficients,G.space.space),G.space)
-(F::ProductFun{S,V,CauchyWeight{O},T1},G::ProductFun{S,V,CauchyWeight{O},T2}) where {S<:UnivariateSpace,V<:UnivariateSpace,O,T1,T2} =
    ProductFun(ProductFun(F.coefficients,F.space.space)-ProductFun(G.coefficients,G.space.space),G.space)

evaluate(f::LowRankFun{S,M,CauchyWeight{O,SV,TT,DD},T},::Colon,::Colon) where {S<:Space,M<:Space,O,SV,TT,T<:Number,DD} =
    error("Not callable.")
evaluate(f::LowRankFun{S,M,CauchyWeight{O,SV,TT,DD},T},x::Number,::Colon) where {S<:Space,M<:Space,O,SV,TT,T<:Number,DD} =
    error("Not callable.")
evaluate(f::LowRankFun{S,M,CauchyWeight{O,SV,TT,DD},T},x::Vector{TTT},::Colon) where {S<:Space,M<:Space,O,SV,TT,T<:Number,TTT<:Number,DD} =
    error("Not callable.")
evaluate(f::LowRankFun{S,M,CauchyWeight{O,SV,TT,DD},T},::Colon,y::Number) where {S<:Space,M<:Space,O,SV,TT,T<:Number,DD} =
    error("Not callable.")
evaluate(f::LowRankFun{S,M,CauchyWeight{O,SV,TT,DD},T},::Colon,y::Vector{TTT}) where {S<:Space,M<:Space,O,SV,TT,T<:Number,TTT<:Number,DD} =
    error("Not callable.")
evaluate(f::LowRankFun{S,M,CauchyWeight{O,SV,TT,DD},T},x,y) where {S<:Space,M<:Space,O,SV,TT,T<:Number,DD} =
    evaluate(f.A,f.B,x,y).*cauchyweight(space(f),x,y)

+(F::LowRankFun{S,M,CauchyWeight{O},T1},G::LowRankFun{S,M,CauchyWeight{O},T2}) where {S<:Space,M<:Space,O,T1<:Number,T2<:Number} =
    LowRankFun([F.A,G.A],[F.B,G.B],F.space)
-(F::LowRankFun{S,M,CauchyWeight{O},T1},G::LowRankFun{S,M,CauchyWeight{O},T2}) where {S<:Space,M<:Space,O,T1<:Number,T2<:Number} =
    LowRankFun([F.A,-G.A],[F.B,G.B],F.space)
