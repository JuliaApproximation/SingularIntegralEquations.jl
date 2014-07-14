
        
## cauchy

import ApproxFun.fromcircle, ApproxFun.tocircle

function cauchyS{N,D<:PeriodicLine}(s::Bool,f::FFun{N,D},z)
    cauchyS(s,FFun(f,Circle()),tocircle(f.domain,z))-    cauchyS(s,FFun(f,Circle()),-1.)
end


function cauchy{N,D<:PeriodicLine}(f::FFun{N,D},z)
    cauchy(FFun(f,Circle()),tocircle(f.domain,z))-    cauchyS(true,FFun(f,Circle()),-1.)
end

function cauchy{N,D<:PeriodicLine}(s::Bool,f::FFun{N,D},z)
    @assert abs(abs(tocircle(f.domain,z))-1.) < 100eps()
    
    cauchyS(s,f,z)
end






