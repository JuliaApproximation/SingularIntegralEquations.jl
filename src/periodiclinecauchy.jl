
        
## cauchy


# function cauchyS(s::Bool,d::PeriodicLine,cfs::Vector,z)
#     cauchyS(s,Circle(),cfs,mappoint(d,Circle(),z))-    cauchyS(s,Circle(),cfs,-1.)
# end
# 
# 
# function cauchy(d::PeriodicLine,cfs::Vector,z)
#     cauchy(Circle(),cfs,mappoint(d,Circle(),z))-    cauchyS(true,Circle(),cfs,-1.)
# end
# 
# function cauchy(s::Bool,d::PeriodicLine,cfs::Vector,z)
#     @assert abs(abs(mappoint(d,Circle(),z))-1.) < 100eps()
#     
#     cauchyS(s,d,cfs,z)
# end

# PeriodicLineSpace doesn't support <: on its own without specifying true/false


function cauchy{S}(f::Fun{PeriodicLineSpace{S}},z::Number)
    g=Fun(f.coefficients,Fourier(Circle()))
    cauchy(g,mappoint(domain(f),Circle(),z))-cauchy(g,-1)
end

function cauchy{S}(s::Bool,f::Fun{PeriodicLineSpace{S}},z::Number)
    g=Fun(f.coefficients,Fourier(Circle()))
    cauchy(s,g,mappoint(domain(f),Circle(),z))-cauchy(g,-1)
end


cauchy{S}(f::Fun{PeriodicLineDirichlet{S}},z::Number)=cauchy(Fun(f,domain(f)),z)
cauchy{S}(s::Bool,f::Fun{PeriodicLineDirichlet{S}},z::Number)=cauchy(s,Fun(f,domain(f)),z)

