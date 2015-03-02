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


function ProductFun{O}(f::Function,cwsp::CauchyWeight{O})
    sp = cwsp.space
    cfs = SymmetricProductFun(f,sp[1],sp[2]).coefficients
    ProductFun{typeof(sp[1]),typeof(sp[2]),typeof(cwsp),eltype(cfs[1])}(cfs,cwsp)
end

evaluate{S<:FunctionSpace,V<:FunctionSpace,O,T}(f::ProductFun{S,V,CauchyWeight{O},T},x::Range,y::Range) = evaluate(f,[x],[y])

function evaluate{S<:FunctionSpace,V<:FunctionSpace,O,T}(f::ProductFun{S,V,CauchyWeight{O},T},x,y)
    ProductFun{S,V,typeof(space(f).space),T}(f.coefficients,space(f).space)[x,y].*cauchyweight(space(f),x,y)
end

+{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O},T},G::ProductFun{S,V,CauchyWeight{O},T}) = ProductFun{S,V,CauchyWeight{O},T}(ProductFun(F.coefficients,F.space.space)+ProductFun(G.coefficients,G.space.space))
-{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O},T},G::ProductFun{S,V,CauchyWeight{O},T}) = ProductFun{S,V,CauchyWeight{O},T}(ProductFun(F.coefficients,F.space.space)-ProductFun(G.coefficients,G.space.space))
