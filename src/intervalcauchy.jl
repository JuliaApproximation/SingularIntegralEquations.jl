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



function stieltjesintervalrecurrence(S,f::AbstractVector,z)
    tol=1./floor(Int,sqrt(length(f)))
    if (abs(real(z))≤1.+tol) && (abs(imag(z))≤tol)
       cfs=stieltjesforward(S,length(f),z)
       dotu(cfs,f)
    else
       cfs=stieltjesbackward(S,z)
       m=min(length(f),length(cfs))
       dotu(cfs[1:m],f[1:m])
    end
end


function stieltjes{D<:Interval}(S::PolynomialSpace{D},f,z::Number)
    if domain(S)==Interval()
        #TODO: check tolerance
        stieltjesintervalrecurrence(Legendre(),coefficients(f,S,Legendre()),z)
    else
        stieltjes(setdomain(S,Interval()),f,tocanonical(S,z))
    end
end

function stieltjes{D<:Interval}(S::PolynomialSpace{D},f,z::Number,s::Bool)
    @assert domain(S)==Interval()

   cfs=stieltjesforward(Legendre(),length(f),z,s)
   dotu(cfs,coefficients(f,S,Legendre()))
end


# Sum over all inverses of fromcanonical, see [Olver,2014]
function stieltjes{SS,L<:Line}(S::MappedSpace{SS,L},f,z,s...)
    if domain(S)==Line()
        stieltjes(S.space,f,tocanonical(S,z),s...) + stieltjes(S.space,f,(-1-sqrt(1+4z.^2))./(2z))
    else
        stieltjes(setdomain(S,Line()),f,mappoint(domain(S),Line(),z),s...)
    end
end
