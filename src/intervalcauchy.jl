import ApproxFun: dotu,SliceOperator


# This solves as a boundary value provblem

jacobiop(S::PolynomialSpace)=transpose(Recurrence(S))
jacobiop(S::JacobiWeight)=jacobiop(S.space)

function stieltjesbackward(S::Space,z::Number)
    J=SliceOperator(jacobiop(S)-z,1,0,1)  # drop first row
    [BasisFunctional(1),
        J]\[stieltjesmoment(S,1,z)]
end


# This solves via forward substitution
function forwardsubstitution!(ret,R,n,μ1,μ2)
    if n≥1
        ret[1]=μ1
    end
    if n≥2
        ret[2]=μ2
    end
    if n≥3
        B=BandedMatrix(R,n-1)
        for k=2:n-1
            ret[k+1]=-(B[k,k-1]*ret[k-1]+B[k,k]*ret[k])/B[k,k+1]
        end
    end
    ret
end

forwardsubstitution(R,n,μ1,μ2)=forwardsubstitution!(Array(promote_type(eltype(R),typeof(μ1),typeof(μ2)),n),R,n,μ1,μ2)

stieltjesforward(sp::Space,n,z,s...)=forwardsubstitution(jacobiop(sp)-z,n,
                        stieltjesmoment(sp,1,z,s...),stieltjesmoment(sp,2,z,s...))



function stieltjesintervalrecurrence(f::Fun,z)
    tol=1./floor(Int,sqrt(length(f)))
    if (abs(real(z))≤1.+tol) && (abs(imag(z))≤tol)
       cfs=stieltjesforward(space(f),length(f),z)
       dotu(cfs,coefficients(f))
    else
       cfs=stieltjesbackward(space(f),z)
       m=min(length(f),length(cfs))
       dotu(cfs[1:m],coefficients(f)[1:m])
    end
end


function stieltjes{PS<:PolynomialSpace}(f::Fun{PS},z::Number)
    if domain(f)==Interval()
        #TODO: check tolerance
        stieltjesintervalrecurrence(Fun(f,Legendre()),z)
    else
        @assert isa(domain(f),Interval)
        stieltjes(setdomain(f,Interval()),tocanonical(f,z))
    end
end

function stieltjes{PS<:PolynomialSpace}(f::Fun{PS},z::Number,s::Bool)
    @assert domain(f)==Interval()

   cfs=stieltjesforward(Legendre(),length(f),z,s)
   dotu(cfs,coefficients(f,Legendre()))
end


# Sum over all inverses of fromcanonical, see [Olver,2014]
function stieltjes{S,L<:Line,T}(f::Fun{MappedSpace{S,L,T}},z,s...)
    if domain(f)==Line()
        p=Fun(f.coefficients,space(f).space)
        stieltjes(p,tocanonical(f,z),s...) + stieltjes(p,(-1-sqrt(1+4z.^2))./(2z))
    else
        p=Fun(f.coefficients,MappedSpace(Line(),space(f).space))
        stieltjes(p,mappoint(domain(f),Line(),z),s...)
    end
end
