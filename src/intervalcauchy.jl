import ApproxFunBase: dotu

# This solves via forward substitution
function forwardsubstitution!(ret,B,μ1,μ2,filter=identity)
    n = length(ret)
    if n≥1
        ret[1] = filter(μ1)
    end
    if n≥2
        ret[2] = filter(μ2)
    end
    if n≥3
        @inbounds for k=2:n-1
            ret[k+1] = filter(-(B[k,k-1]*ret[k-1]+B[k,k]*ret[k])/B[k,k+1])
        end
    end
    ret
end

forwardsubstitution(R,n,μ1,μ2) =
    forwardsubstitution!(Array{promote_type(eltype(R),typeof(μ1),typeof(μ2))}(undef,n),R,μ1,μ2)



# This solves as a boundary value provblem
stieltjesbackward(S::Space,z::Number) = JacobiZ(S,z)\[stieltjesmoment(S,0,z)]


stieltjesbackward!(ret,S::Space,z::Number) = ret[:] = stieltjesbackward(S,z)[1:length(ret)]


stieltjesforward(sp::Space,n,z) = forwardsubstitution(JacobiZ(sp,undirected(z)),n,
                                                      stieltjesmoment(sp,0,z),
                                                      stieltjesmoment(sp,1,z))

stieltjesforward!(ret,sp::Space,z,filter=identity) = forwardsubstitution!(ret,JacobiZ(sp,undirected(z)),
                                                    stieltjesmoment(sp,0,z),
                                                    stieltjesmoment(sp,1,z),filter)

hilbertforward(sp::Space,n,z) = forwardsubstitution(JacobiZ(sp,z),n,
                                                    hilbertmoment(sp,0,z),
                                                    hilbertmoment(sp,1,z))



function stieltjesmoment!(ret,S::PolynomialSpace{<:IntervalOrSegment},z,filter=identity)
    if domain(S) == ChebyshevInterval()
        n = length(ret)
        tol = 1/floor(Int,sqrt(n))
        if (abs(real(z)) ≤ 1+tol) && (abs(imag(z)) ≤ tol)
            cfs = stieltjesforward!(ret,S,z,filter)
        else
            cfs = stieltjesbackward!(ret,S,z)
        end

        ret
    else
        stieltjesmoment!(ret,setdomain(S,ChebyshevInterval()),mobius(S,z),filter)
    end
end


function stieltjesintervalrecurrence(S,f::AbstractVector,z)
    tol=1/floor(Int,sqrt(length(f)))
    if isinf(z)
        zero(promote_type(typeof(z), eltype(f)))
    elseif (abs(real(z)) ≤ 1+tol) && (abs(imag(z)) ≤ tol)
        cfs = stieltjesforward(S,length(f),z)
        dotu(cfs,f)
    else
        cfs = stieltjesbackward(S,undirected(z))
        dotu(cfs,f)
    end
end

stieltjesintervalrecurrence(S,f::AbstractVector,z::AbstractArray) =
    reshape(promote_type(eltype(f),eltype(z))[ stieltjesintervalrecurrence(S,f,z[i]) for i in eachindex(z) ], size(z))


function stieltjes(S::PolynomialSpace{<:IntervalOrSegment},f,z::Number)
    if domain(S)==ChebyshevInterval()
        #TODO: check tolerance
        stieltjesintervalrecurrence(Legendre(),coefficients(f,S,Legendre()),z)
    else
        stieltjes(setdomain(S,ChebyshevInterval()),f,mobius(S,z))
    end
end

function hilbert(S::PolynomialSpace{<:IntervalOrSegment},f,z::Number)
    if domain(S) == ChebyshevInterval()
        L = Legendre(domain(S))
        cfs = hilbertforward(L,length(f),z)
        dotu(cfs,coefficients(f, S, L))
    else
        hilbert(setdomain(S,ChebyshevInterval()),f,mobius(S,z))
    end
end




# Sum over all inverses of fromcanonical, see [Olver,2014]
function stieltjes(S::Space{<:Line},f,z)
    if domain(S)==Line()
        # TODO: rename tocanonical
        stieltjes(setcanonicaldomain(S),f,tocanonical(S,z)) +
            stieltjes(setcanonicaldomain(S),f,(-1-sqrt(1+4z^2))/(2z))
    else
        stieltjes(setdomain(S,Line()),f,mappoint(domain(S),Line(),z))
    end
end



## log kernel



function logkernel(S::PolynomialSpace{<:IntervalOrSegment},v,z::Number)
    if domain(S) == ChebyshevInterval()
        DS=JacobiWeight(1,1,Jacobi(1,1))
        D=Derivative(DS)[2:end,:]

        f=Fun(Fun(S,v),Legendre())  # convert to Legendre expansion
        u=D\(f|(2:∞))   # find integral, dropping first coefficient of f

        (f.coefficients[1]*logabslegendremoment(z) +
            real(stieltjes(Fun(u,Legendre()),z+0im)))/π
    else
        Mp=abs(fromcanonicalD(S,0))
        Mp*logkernel(setcanonicaldomain(S),v,mobius(S,z)) +
            linesum(Fun(S,v))*log(Mp)/π
    end
end
