module RiemannHilbert
    using Base, ApproxFun

export CauchyOperator, cauchy
import ApproxFun
import ApproxFun.PeriodicDomain
import ApproxFun.BandedShiftOperator
import ApproxFun.bandrange

type CauchyOperator{D<:PeriodicDomain} <: BandedShiftOperator{Complex{Float64}}
    sign::Bool
    domain::D
end

function CauchyOperator(s::Integer,d)
    @assert abs(s) == 1
    CauchyOperator(s==1,d)
end

CauchyOperator(s)=CauchyOperator(s,Circle())

bandrange(::CauchyOperator)=0:0
domainspace(D::CauchyOperator)=FourierSpace(D.domain)
rangespace(D::CauchyOperator)=FourierSpace(D.domain)

function cauchy_pos_addentries!(A::ShiftArray,kr::Range1)
    for k=max(0,kr[1]):kr[end]
        A[k,0]+=1.
    end
    
    A
end

function cauchy_neg_addentries!(A::ShiftArray,kr::Range1)
    for k=kr[1]:min(-1,kr[end])
        A[k,0]+=-1.
    end
    
    A
end

ApproxFun.addentries!(C::CauchyOperator,A::ShiftArray,kr::Range1)=C.sign?
    cauchy_pos_addentries!(A,kr):
    cauchy_neg_addentries!(A,kr)
        
        
## cauchy

function cauchyS{N,D<:Circle}(s::Bool,f::FFun{N,D},z)
    @assert f.domain.center == 0 && f.domain.radius == 1
    
    ret=zero(Complex{Float64})
    
    if s
        zm = one(Complex{Float64})
        
        for k=0:lastindex(f.coefficients)
            ret += f.coefficients[k]*zm
            zm *= z
        end
    else
        z=1./z
        zm = z

        for k=-1:-1:firstindex(f.coefficients)
            ret -= f.coefficients[k]*zm
            zm *= z
        end
    end
    
    ret
end


function cauchy{N,D<:Circle}(f::FFun{N,D},z)
    @assert f.domain.center == 0 && f.domain.radius == 1
    
    cauchyS(abs(z) < 1,f,z)
end

function cauchy{N,D<:Circle}(s::Bool,f::FFun{N,D},z)
    @assert f.domain.center == 0 && f.domain.radius == 1
    @assert abs(abs(z)-1.) < 100eps()
    
    cauchyS(s,f,z)
end

function cauchy(s::Integer,f,z)
    @assert abs(s) == 1
    
    cauchy(s==1,f,z)
end

end #module


