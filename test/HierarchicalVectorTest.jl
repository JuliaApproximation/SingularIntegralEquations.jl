#
# test HierarchicalVector
#
using ApproxFun, SingularIntegralEquations, Base.Test

a = rand(5);b = rand(5);v = HierarchicalVector((a,b)); V = HierarchicalVector((v,v));r = rand(10);

@test isa(v,HierarchicalVector)

@test isa(V,HierarchicalVector)

@test norm(v-[a;b]) < 10eps()

@test full(1+2v) == 1+2full(v)

@test v.^2 == v.*v

@test norm((v+r)⊖(r⊕v)) < 10eps()

@test 2(v⋅v) == V⋅V

@test 1+nlevels(v) == nlevels(V)

@test abs(cumsum(v)[end]-sum(v)) < 10eps()

@test partition(V) == (v,v)

w = v+im*v
u = copy(w)

@test conj!(w) == conj(u)

@test 1+zero(v) == ones(v)
