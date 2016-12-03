import ApproxFun: dotu

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

forwardsubstitution(R,n,μ1,μ2) =
    forwardsubstitution!(Array(promote_type(eltype(R),typeof(μ1),typeof(μ2)),n),R,n,μ1,μ2)



# This solves as a boundary value provblem
stieltjesbackward(S::Space,z::Number) = JacobiZ(S,z)\[stieltjesmoment(S,0,z)]


stieltjesforward(sp::Space,n,z) = forwardsubstitution(JacobiZ(sp,value(z)),n,
                                                            stieltjesmoment(sp,0,z),
                                                            stieltjesmoment(sp,1,z))

hilbertforward(sp::Space,n,z) = forwardsubstitution(JacobiZ(sp,z),n,
                                                            hilbertmoment(sp,0,z),
                                                            hilbertmoment(sp,1,z))


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

stieltjesintervalrecurrence(S,f::AbstractVector,z::AbstractArray) =
    reshape(promote_type(eltype(f),eltype(z))[ stieltjesintervalrecurrence(S,f,z[i]) for i in eachindex(z) ], size(z))


function stieltjes{D<:Segment}(S::PolynomialSpace{D},f,z::Number)
    if domain(S)==Segment()
        #TODO: check tolerance
        stieltjesintervalrecurrence(Legendre(),coefficients(f,S,Legendre()),z)
    else
        stieltjes(setdomain(S,Segment()),f,mobius(S,z))
    end
end

function hilbert{D<:Segment}(S::PolynomialSpace{D},f,z::Number)
    if domain(S)==Segment()
        cfs = hilbertforward(S,length(f),z)
        dotu(cfs,f)
    else
        hilbert(setdomain(S,Segment()),f,mobius(S,z))
    end
end




# Sum over all inverses of fromcanonical, see [Olver,2014]
function stieltjes{SS,L<:Line}(S::Space{SS,L},f,z)
    if domain(S)==Line()
        # TODO: rename tocanonical
        stieltjes(setcanonicaldomain(S),f,tocanonical(S,z)) +
            stieltjes(setcanonicaldomain(S),f,(-1-sqrt(1+4z.^2))./(2z))
    else
        stieltjes(setdomain(S,Line()),f,mappoint(domain(S),Line(),z))
    end
end



## log kernel



function logkernel{DD<:Segment}(S::PolynomialSpace{DD},v,z::Number)
    if domain(S) == Segment()
        DS=JacobiWeight(1,1,Jacobi(1,1))
        D=Derivative(DS)[2:end,:]

        f=Fun(Fun(S,v),Legendre())  # convert to Legendre expansion
        u=D\(f|(2:∞))   # find integral, dropping first coefficient of f

        (f.coefficients[1]*logabslegendremoment(z) + real(stieltjes(Fun(u,Legendre()),z+0im)))/π
    else
        Mp=abs(fromcanonicalD(S,0))
        Mp*logkernel(setcanonicaldomain(S),v,mobius(S,z))+linesum(Fun(S,v))*log(Mp)/π
    end
end

for FUNC in (:logkernel,:stieltjes)
    @eval $FUNC{D<:Segment}(S::PolynomialSpace{D},f,z::AbstractArray) =
        map(x->$FUNC(S,f,x),z)
end
