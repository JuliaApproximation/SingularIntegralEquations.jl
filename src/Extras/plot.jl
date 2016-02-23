const colours = Dict(0=>:black,1=>:red,2=>:green,3=>:blue,4=>:cyan,5=>:magenta,6=>:yellow,7=>:orange)

## HierarchicalDomain

using Plots

Plots.plot(H::HierarchicalDomain;grid=true,kwds...) = plot!(plot(grid=grid),H;kwds...)
Plots.plot!(H::HierarchicalDomain;kwds...) = plot!(current(),H;kwds...)

function Plots.plot!(plt::Plots.Plot,H::HierarchicalDomain;kwds...)
    H1,H2 = partition(H)
    plot!(plt,H1;kwds...)
    plot!(plt,H2;kwds...)
end
