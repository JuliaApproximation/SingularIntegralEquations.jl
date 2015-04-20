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
## This will change when we switch to ChebyshevDirichlet bases

for Func in (:ProductFun,:convolutionProductFun)
    @eval begin
        function $Func{O}(f::Function,cwsp::CauchyWeight{O};kwds...)
            F = domain(cwsp[1]) == domain(cwsp[2]) ? $Func(f,cwsp[1],cwsp[2];kwds...) : $Func((x,y)->f(x,y)*cauchyweight(O,x,y),cwsp[1],cwsp[2];kwds...)
            return ProductFun(F.coefficients,cwsp)
        end
    end
end

function LowRankFun{O}(f::Function,cwsp::CauchyWeight{O};kwds...)
    F = domain(cwsp[1]) == domain(cwsp[2]) ? LowRankFun(f,cwsp[1],cwsp[2];kwds...) : LowRankFun((x,y)->f(x,y)*cauchyweight(O,x,y),cwsp[1],cwsp[2];kwds...)
    LowRankFun(F.A,F.B,cwsp)
end

## Definite (Line) Integration over BivariateFuns in a CauchyWeight space

## TODO: for different domains, should be OffOp instead of ⨍
## This will change when we switch to ChebyshevDirichlet bases

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

evaluate{S<:FunctionSpace,V<:FunctionSpace,O,T}(f::ProductFun{S,V,CauchyWeight{O,(S,V),T},T},x::Range,y::Range) = evaluate(f,[x],[y])
evaluate{S<:FunctionSpace,V<:FunctionSpace,O,T}(f::ProductFun{S,V,CauchyWeight{O,(S,V),T},T},x,y) = ProductFun{S,V,typeof(space(f).space),T}(f.coefficients,space(f).space)[x,y].*cauchyweight(space(f),x,y)

+{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O,(S,V),T},T},G::ProductFun{S,V,CauchyWeight{O,(S,V),T},T}) = ProductFun(ProductFun(F.coefficients,F.space.space)+ProductFun(G.coefficients,G.space.space),G.space)
-{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O,(S,V),T},T},G::ProductFun{S,V,CauchyWeight{O,(S,V),T},T}) = ProductFun(ProductFun(F.coefficients,F.space.space)-ProductFun(G.coefficients,G.space.space),G.space)
