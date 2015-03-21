# CauchyWeight

export CauchyWeight

immutable CauchyWeight{O} <: AbstractProductSpace
    space::AbstractProductSpace
    CauchyWeight(space) = new(space)
end

order{O}(::CauchyWeight{O}) = O
domain(C::CauchyWeight)=domain(C.space)

cauchyweight(O,x,y) = O == 0 ? logabs(y-x)/π : (y-x).^(-O)/π
cauchyweight{O}(C::CauchyWeight{O},x,y) = cauchyweight(O,tocanonical(C,x,y)...)

Base.getindex{BT,S,V,O,T}(B::Operator{BT},f::ProductFun{S,V,CauchyWeight{O},T}) = PlusOperator(BandedOperator{promote_type(BT,T)}[f.coefficients[i]*B[Fun([zeros(promote_type(BT,T),i-1),one(promote_type(BT,T))],f.space.space[2])] for i=1:length(f.coefficients)])


function ProductFun{O}(f::Function,cwsp::CauchyWeight{O};method::Symbol=:convolution)
    sp = cwsp.space
    cfs = ProductFun(f,sp[1],sp[2];method=method).coefficients
    ProductFun{typeof(sp[1]),typeof(sp[2]),typeof(cwsp),eltype(cfs[1])}(cfs,cwsp)
end
ProductFun{O}(F::ProductFun,cwsp::CauchyWeight{O}) = ProductFun{typeof(cwsp.space[1]),typeof(cwsp.space[2]),typeof(cwsp),eltype(F)}(F.coefficients,cwsp)

evaluate{S<:FunctionSpace,V<:FunctionSpace,O,T}(f::ProductFun{S,V,CauchyWeight{O},T},x::Range,y::Range) = evaluate(f,[x],[y])
evaluate{S<:FunctionSpace,V<:FunctionSpace,O,T}(f::ProductFun{S,V,CauchyWeight{O},T},x,y) = ProductFun{S,V,typeof(space(f).space),T}(f.coefficients,space(f).space)[x,y].*cauchyweight(space(f),x,y)

+{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O},T},G::ProductFun{S,V,CauchyWeight{O},T}) = ProductFun(ProductFun(F.coefficients,F.space.space)+ProductFun(G.coefficients,G.space.space),G.space)
-{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O},T},G::ProductFun{S,V,CauchyWeight{O},T}) = ProductFun(ProductFun(F.coefficients,F.space.space)-ProductFun(G.coefficients,G.space.space),G.space)

##TODO: should check domainspace and rangespace and determine if it should be Op or OffOp.
Base.getindex{S,V1,V2,O,T1,T2}(⨍::DefiniteIntegral{V1,T1},f::ProductFun{S,V2,CauchyWeight{O},T2}) = Hilbert(⨍.domainspace,O)[f]
Base.getindex{S,V1,V2,O,T1,T2}(⨍::DefiniteLineIntegral{V1,T1},f::ProductFun{S,V2,CauchyWeight{O},T2}) = SingularIntegral(⨍.domainspace,O)[f]
