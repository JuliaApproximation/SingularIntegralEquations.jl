# CauchyWeight enriches a TensorSpace with a singular diagonal

export CauchyWeight

immutable CauchyWeight{O,SV,T} <: AbstractProductSpace{SV,T,2}
    space::TensorSpace{SV,T,2}
end

CauchyWeight{SV,T}(space::TensorSpace{SV,T,2},O) = CauchyWeight{O,SV,T}(space)

order{O}(C::CauchyWeight{O}) = O
domain(C::CauchyWeight)=domain(C.space)
Base.getindex(C::CauchyWeight,k::Integer)=C.space[k]
ApproxFun.columnspace(C::CauchyWeight,::)=C[1]
Base.getindex{O,PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}(C::CauchyWeight{O,(PWS1,PWS2)},i::Integer,j::Integer)=CauchyWeight(C[1][i]⊗C[2][j],O)
Base.transpose{O}(C::CauchyWeight{O}) = CauchyWeight(transpose(C.space),O)

cauchyweight(O,x,y) = O == 0 ? logabs(y-x)/π : (y-x).^(-O)/π
cauchyweight{O}(C::CauchyWeight{O},x,y) = cauchyweight(O,tocanonical(C,x,y)...)






## BivariateFun constructors in a CauchyWeight space

## TODO: for different domains, there should not be (x,y)->f(x,y)*cauchyweight(O,x,y)
## This will change for v0.0.2 with the switch to ChebyshevDirichlet{1,1} bases

for Func in (:ProductFun,:convolutionProductFun)
    @eval begin
        function $Func{O}(f::Function,cwsp::CauchyWeight{O};kwds...)
            F = domain(cwsp[1]) == domain(cwsp[2]) ? $Func(f,cwsp[1],cwsp[2];kwds...) : $Func((x,y)->f(x,y)*cauchyweight(O,x,y),cwsp[1],cwsp[2];kwds...)
            return ProductFun(F.coefficients,cwsp)
        end
    end
end

function LowRankFun{O}(f::Function,cwsp::CauchyWeight{O};retmax::Bool=false,kwds...)
    if retmax
        F,maxabsf = domain(cwsp[1]) == domain(cwsp[2]) ? LowRankFun(f,cwsp[1],cwsp[2];retmax=retmax,kwds...) : LowRankFun((x,y)->f(x,y)*cauchyweight(O,x,y),cwsp[1],cwsp[2];retmax=retmax,kwds...)
        LowRankFun(F.A,F.B,cwsp),maxabsf
    else
        F = domain(cwsp[1]) == domain(cwsp[2]) ? LowRankFun(f,cwsp[1],cwsp[2];kwds...) : LowRankFun((x,y)->f(x,y)*cauchyweight(O,x,y),cwsp[1],cwsp[2];kwds...)
        LowRankFun(F.A,F.B,cwsp)
    end
end

## Definite (Line) Integration over BivariateFuns in a CauchyWeight space

## TODO: for different domains, should be OffOp instead of ⨍
## This will change for v0.0.2 with the switch to ChebyshevDirichlet{1,1} bases

for (Func,Op) in ((:DefiniteIntegral,:Hilbert),(:DefiniteLineIntegral,:SingularIntegral))
    @eval begin
        function Base.getindex{S,V,O,T,V1,T1,T2}(⨍::$Func{V1,T1},f::ProductFun{S,V,CauchyWeight{O,(S,V),T2},T})
            if domain(f.space[1]) == domain(f.space[2])
                $Op(⨍.domainspace,O)[f]
            else
                ⨍[ProductFun(f.coefficients,f.space.space)]
            end
        end
        function Base.getindex{S,M,O,T,V,V1,T1,T2}(⨍::$Func{V1,T1},f::LowRankFun{S,M,CauchyWeight{O,(S,M),T2},T,V})
            if domain(f.space[1]) == domain(f.space[2])
                $Op(⨍.domainspace,O)[f]
            else
                ⨍[LowRankFun(f.A,f.B,f.space.space)]
            end
        end
    end
end




## Evaluation of bivariate functions in a CauchyWeight space

evaluate{S<:UnivariateSpace,V<:UnivariateSpace,O,T1,T2}(f::ProductFun{S,V,CauchyWeight{O,(S,V),T1},T2},x::Range,y::Range) = evaluate(f,[x],[y])
evaluate{S<:UnivariateSpace,V<:UnivariateSpace,O,T1,T2}(f::ProductFun{S,V,CauchyWeight{O,(S,V),T1},T2},x,y) = evaluate(ProductFun(f.coefficients,space(f).space),x,y).*cauchyweight(space(f),x,y)

+{S<:UnivariateSpace,V<:UnivariateSpace,O,T1,T2}(F::ProductFun{S,V,CauchyWeight{O},T1},G::ProductFun{S,V,CauchyWeight{O},T2}) = ProductFun(ProductFun(F.coefficients,F.space.space)+ProductFun(G.coefficients,G.space.space),G.space)
-{S<:UnivariateSpace,V<:UnivariateSpace,O,T1,T2}(F::ProductFun{S,V,CauchyWeight{O},T1},G::ProductFun{S,V,CauchyWeight{O},T2}) = ProductFun(ProductFun(F.coefficients,F.space.space)-ProductFun(G.coefficients,G.space.space),G.space)

evaluate{S<:FunctionSpace,M<:FunctionSpace,O,T<:Number,V<:Number}(f::LowRankFun{S,M,CauchyWeight{O},T,V},::Colon,::Colon) = error("Not callable.")
evaluate{S<:FunctionSpace,M<:FunctionSpace,O,T<:Number,V<:Number}(f::LowRankFun{S,M,CauchyWeight{O},T,V},x,::Colon) = error("Not callable.")
evaluate{S<:FunctionSpace,M<:FunctionSpace,O,T<:Number,V<:Number}(f::LowRankFun{S,M,CauchyWeight{O},T,V},::Colon,y) = error("Not callable.")
evaluate{S<:FunctionSpace,M<:FunctionSpace,O,T<:Number,V<:Number}(f::LowRankFun{S,M,CauchyWeight{O},T,V},x,y) = evaluate(f.A,f.B,x,y).*cauchyweight(space(f),x,y)

+{S<:FunctionSpace,M<:FunctionSpace,O,T1<:Number,V1<:Number,T2<:Number,V2<:Number}(F::LowRankFun{S,M,CauchyWeight{O},T1,V1},G::LowRankFun{S,M,CauchyWeight{O},T2,V2}) = LowRankFun([F.A,G.A],[F.B,G.B],F.space)
-{S<:FunctionSpace,M<:FunctionSpace,O,T1<:Number,V1<:Number,T2<:Number,V2<:Number}(F::LowRankFun{S,M,CauchyWeight{O},T1,V1},G::LowRankFun{S,M,CauchyWeight{O},T2,V2}) = LowRankFun([F.A,-G.A],[F.B,G.B],F.space)
