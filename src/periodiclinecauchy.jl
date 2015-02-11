
        
## cauchy


function cauchyS(s::Bool,d::PeriodicLine,cfs::Vector,z)
    cauchyS(s,Circle(),cfs,mappoint(d,Circle(),z))-    cauchyS(s,Circle(),cfs,-1.)
end


function cauchy(d::PeriodicLine,cfs::Vector,z)
    cauchy(Circle(),cfs,mappoint(d,Circle(),z))-    cauchyS(true,Circle(),cfs,-1.)
end

function cauchy(s::Bool,d::PeriodicLine,cfs::Vector,z)
    @assert abs(abs(mappoint(d,Circle(),z))-1.) < 100eps()
    
    cauchyS(s,d,f,z)
end






