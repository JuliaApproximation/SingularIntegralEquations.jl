#
# test HierarchicalVector
#
using ApproxFun, SingularIntegralEquations, Base.Test

a = rand(5);b = rand(5);v = HierarchicalVector((a,b)); V = HierarchicalVector((v,a,v));r = rand(10);

@test isa(v,HierarchicalVector)

@test isa(V,HierarchicalVector)

@test norm(v-[a;b]) < 10eps()

@test full(1+2v) == 1+2full(v)

@test v.^2 == v.*v

@test norm((v+r)⊖(r⊕v)) < 10eps()

@test v⋅v+a⋅a+v⋅v == V⋅V == (V'*V)[1]

@test 1+nlevels(v) == nlevels(V)

@test abs(cumsum(v)[end]-sum(v)) < 10eps()

@test partition(V) == (v,a,v)

@test promote_rule(HierarchicalVector{Float32,nchildren(V)},typeof(V)) == typeof(V)

@test promote_rule(HierarchicalVector{Complex128,nchildren(V)},typeof(V)) == HierarchicalVector{Complex128,nchildren(V)}

@test norm(convert(HierarchicalVector{Complex128,nchildren(V)},V)-V) == 0

x = v+im*r
y = v⊕im*r
@test norm(x-y) == 0
w = v+im*v
u = copy(w)

@test conj!(w) == conj(u)

@test 1+zero(v) == ones(v)
