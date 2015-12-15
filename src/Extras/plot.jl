const colours = Dict(0=>"k",1=>"r",2=>"g",3=>"b",4=>"c",5=>"m",6=>"y",7=>"orange")

## HierarchicalDomain

function ApproxFun.plot(H::HierarchicalDomain;kwds...)
    H1,H2 = partition(H)
    ApproxFun.plot(H1;kwds...)
    ApproxFun.plot(H2;kwds...)
end
