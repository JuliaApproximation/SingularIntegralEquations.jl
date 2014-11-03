
        
## cauchy

import ApproxFun.fromcircle, ApproxFun.tocircle

function cauchyS(s::Bool,d::PeriodicLine,cfs::Vector,z)
    cauchyS(s,Circle(),cfs,tocircle(d,z))-    cauchyS(s,Circle(),cfs,-1.)
end


function cauchy(d::PeriodicLine,cfs::Vector,z)
    cauchy(Circle(),cfs,tocircle(d,z))-    cauchyS(true,Circle(),cfs,-1.)
end

function cauchy(s::Bool,d::PeriodicLine,cfs::Vector,z)
    @assert abs(abs(tocircle(d,z))-1.) < 100eps()
    
    cauchyS(s,d,f,z)
end






