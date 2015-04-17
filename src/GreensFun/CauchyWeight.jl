# CauchyWeight enriches a TensorSpace with a singular diagonal

export CauchyWeight

immutable CauchyWeight{O,SV,T} <: AbstractProductSpace{SV,T,2}
    space::TensorSpace{SV,T,2}
end

CauchyWeight{SV,T}(space::TensorSpace{SV,T,2},O) = CauchyWeight{O,SV,T}(space)

order{O}(C::CauchyWeight{O}) = O
domain(C::CauchyWeight)=domain(C.space)
Base.getindex(C::CauchyWeight,k::Integer)=C.space[k]

cauchyweight(O,x,y) = O == 0 ? logabs(y-x)/π : (y-x).^(-O)/π
cauchyweight{O}(C::CauchyWeight{O},x,y) = cauchyweight(O,tocanonical(C,x,y)...)








## BivariateFun constructors in a CauchyWeight space

##TODO: for different domains, there should not be (x,y)->f(x,y)*cauchyweight(O,x,y)

function ProductFun{O}(f::Function,cwsp::CauchyWeight{O};method::Symbol=:convolution,tol=eps())
    cfs = domain(cwsp[1]) == domain(cwsp[2]) ? ProductFun(f,cwsp[1],cwsp[2];method=method,tol=tol).coefficients : ProductFun((x,y)->f(x,y)*cauchyweight(O,x,y),cwsp[1],cwsp[2];method=method,tol=tol).coefficients
    ProductFun(cfs,cwsp)
end



## Definite (Line) Integration over BivariateFuns in a CauchyWeight space

##TODO: for different domains, should be OffOp instead of ⨍

function Base.getindex{S,V,O,T,V1,T1}(⨍::DefiniteIntegral{V1,T1},f::ProductFun{S,V,CauchyWeight{O},T})
    if domain(f.space[1]) == domain(f.space[2])
        Hilbert(⨍.domainspace,O)[f]
    else
        ⨍[ProductFun(f.coefficients,f.space.space)]
    end
end
function Base.getindex{S,V,O,T,V1,T1}(⨍::DefiniteLineIntegral{V1,T1},f::ProductFun{S,V,CauchyWeight{O},T})
    if domain(f.space[1]) == domain(f.space[2])
        SingularIntegral(⨍.domainspace,O)[f]
    else
        ⨍[ProductFun(f.coefficients,f.space.space)]
    end
end



## Evaluation of bivariate functions in a CauchyWeight space

evaluate{S<:FunctionSpace,V<:FunctionSpace,O,T}(f::ProductFun{S,V,CauchyWeight{O,(S,V),T},T},x::Range,y::Range) = evaluate(f,[x],[y])
evaluate{S<:FunctionSpace,V<:FunctionSpace,O,T}(f::ProductFun{S,V,CauchyWeight{O,(S,V),T},T},x,y) = ProductFun{S,V,typeof(space(f).space),T}(f.coefficients,space(f).space)[x,y].*cauchyweight(space(f),x,y)

+{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O,(S,V),T},T},G::ProductFun{S,V,CauchyWeight{O,(S,V),T},T}) = ProductFun(ProductFun(F.coefficients,F.space.space)+ProductFun(G.coefficients,G.space.space),G.space)
-{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O,(S,V),T},T},G::ProductFun{S,V,CauchyWeight{O,(S,V),T},T}) = ProductFun(ProductFun(F.coefficients,F.space.space)-ProductFun(G.coefficients,G.space.space),G.space)



## Default composition of an Operator with a BivariateFun in a CauchyWeight space. No longer required because of Base.getindex(::CauchyWeight)
#Base.getindex{BT,S,V,O,T}(B::Operator{BT},f::ProductFun{S,V,CauchyWeight{O},T}) = mapreduce(i->f.coefficients[i]*B[Fun([zeros(promote_type(BT,T),i-1),one(promote_type(BT,T))],f.space.space[2])],+,1:length(f.coefficients))
# useless?
#ProductFun{SV,O,T}(F::ProductFun,cwsp::CauchyWeight{SV,O,T}) = ProductFun{typeof(cwsp.space[1]),typeof(cwsp.space[2]),typeof(cwsp),eltype(F)}(F.coefficients,cwsp)
#Base.getindex{S,V,O,T,V1,T1}(⨍::DefiniteLineIntegral{V1,T1},f::ProductFun{S,V,CauchyWeight{O},T}) = SingularIntegral(⨍.domainspace,O)[f]
