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
