import ApproxFun: dotu


# This solves as a boundary value provblem

stieltjesbackward(S::Space,z::Number) = JacobiZ(S,z)\[stieltjesmoment(S,0,z)]

# This solves via forward substitution
function forwardsubstitution!(ret,B,n,μ1,μ2)
    if n≥1
        ret[1]=μ1
    end
    if n≥2
        ret[2]=μ2
    end
    if n≥3
        for k=2:n-1
            ret[k+1]=-(B[k,k-1]*ret[k-1]+B[k,k]*ret[k])/B[k,k+1]
        end
    end
    ret
end

forwardsubstitution(R,n,μ1,μ2)=forwardsubstitution!(Array(promote_type(eltype(R),typeof(μ1),typeof(μ2)),n),R,n,μ1,μ2)

stieltjesforward(sp::Space,n,z,s...)=forwardsubstitution(JacobiZ(sp,z),n,
                        stieltjesmoment(sp,0,z,s...),stieltjesmoment(sp,1,z,s...))


function stieltjesintervalrecurrence(S,f::AbstractVector,z)
    tol=1./floor(Int,sqrt(length(f)))
    if (abs(real(z))≤1.+tol) && (abs(imag(z))≤tol)
        cfs = stieltjesforward(S,length(f),z)
        dotu(cfs,f)
    else
        cfs = stieltjesbackward(S,z)
        dotu(cfs,f)
    end
end
stieltjesintervalrecurrence(S,f::AbstractVector,z::AbstractArray) = reshape(promote_type(eltype(f),eltype(z))[ stieltjesintervalrecurrence(S,f,z[i]) for i in eachindex(z) ], size(z))


function stieltjes{D<:Interval}(S::PolynomialSpace{D},f,z::Number)
    if domain(S)==Interval()
        #TODO: check tolerance
        stieltjesintervalrecurrence(Legendre(),coefficients(f,S,Legendre()),z)
    else
        stieltjes(setdomain(S,Interval()),f,mobius(S,z))
    end
end

function stieltjes{D<:Interval}(S::PolynomialSpace{D},f,z::Number,s::Bool)
    @assert domain(S)==Interval()

   cfs=stieltjesforward(Legendre(),length(f),z,s)
   dotu(cfs,coefficients(f,S,Legendre()))
end


# Sum over all inverses of fromcanonical, see [Olver,2014]
function stieltjes{SS,L<:Line}(S::Space{SS,L},f,z,s...)
    if domain(S)==Line()
        # TODO: rename tocanonical
        stieltjes(setcanonicaldomain(S),f,tocanonical(S,z),s...) +
            stieltjes(setcanonicaldomain(S),f,(-1-sqrt(1+4z.^2))./(2z))
    else
        stieltjes(setdomain(S,Line()),f,mappoint(domain(S),Line(),z),s...)
    end
end
