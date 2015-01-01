
import ApproxFun:mappoint
        
## cauchy

function cauchyS(s::Bool,cfs::Vector,z::Number)    
    ret=zero(Complex{Float64})
    
    if s
        zm = one(Complex{Float64})
        
        #odd coefficients are pos
        @simd for k=1:2:length(cfs)
            @inbounds ret += cfs[k]*zm
            zm *= z
        end
    else
        z=1./z
        zm = z

        #even coefficients are neg
        @simd for k=2:2:length(cfs)
            @inbounds ret -= cfs[k]*zm
            zm *= z
        end
    end
    
    ret
end

cauchyS(s::Bool,d::Circle,cfs::Vector,z::Number)=cauchyS(s,cfs,mappoint(d,Circle(),z))


function cauchy(d::Circle,cfs::Vector,z::Number)
    z=mappoint(d,Circle(),z)
    cauchyS(abs(z) < 1,cfs,z)
end

cauchy(d::Circle,cfs::Vector,z::Vector)=[cauchy(d,cfs,zk) for zk in z]

function cauchy(s::Bool,d::Circle,cfs::Vector,z::Number)
    @assert in(z,d)
    
    cauchyS(s,d,cfs,z)
end



cauchy(s::Bool,f::Fun{Laurent},z::Number)=cauchy(s,domain(f),coefficients(f),z)
cauchy(f::Fun{Laurent},z::Number)=cauchy(domain(f),coefficients(f),z)






## mapped Cauchy

function cauchy(f::Fun{CurveSpace{Laurent}},z::Number)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()
    # subtract out value at infinity, determined by the fact that leading term is poly
    sum(cauchy(fcirc,complexroots(c.curve-z)))-div(length(c.curve),2)*cauchy(fcirc,0.)
end

