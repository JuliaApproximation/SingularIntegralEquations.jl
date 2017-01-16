const colours = Dict(0=>:black,1=>:red,2=>:green,3=>:blue,4=>:cyan,5=>:magenta,6=>:yellow,7=>:orange)

## HierarchicalDomain

@recipe function f(H::HierarchicalDomain)
    H1,H2 = partition(H)
    @series begin H1 end
    @series begin H2 end
end
