import ApproxFun: dotu,SliceOperator


# This solves as a boundary value provblem
function cauchylegendrebackward(z::Number)
    J=SliceOperator(Recurrence(Jacobi(0.,0.)).'-z,1,0,1)  # drop first row
    [BasisFunctional(1),
        J]\[(log(z-1)-log(z+1))/(2π*im)]
end


# This solves via forward substitution
function forwardsubstitution(R,n,μ1,μ2)
    T=promote_type(eltype(R),typeof(μ1),typeof(μ2))
    if n==1
        T[μ1]
    elseif n==2
        T[μ1,μ2]
    else
        B=BandedMatrix(R,n-1)
        ret=Array(T,n)
        ret[1]=μ1
        ret[2]=μ2
        for k=2:n-1
            ret[k+1]=-(B[k,k-1]*ret[k-1]+B[k,k]*ret[k])/B[k,k+1]
        end
        ret
    end
end

cauchylegendreforward(n,z)=forwardsubstitution(Recurrence(Jacobi(0.,0.)).'-z,n,
                        (log(z-1)-log(z+1))/(2π*im),(2-z*log(1+z)+z*log(z-1))/(2π*im))

cauchylegendreforward(s::Bool,n,z)=forwardsubstitution(Recurrence(Jacobi(0.,0.)).'-z,n,
                        (log(1-z)+(s?1:-1)*π*im-log(z+1))/(2π*im),(2-z*log(1+z)+z*log(1-z) + (s?1:-1)*π*im*z)/(2π*im))


function cauchy(f::Fun{Jacobi},z::Number)
    @assert space(f).a==0 && space(f).b==0
    @assert domain(f)==Interval()
    #TODO: check tolerance
    tol=1./ifloor(Int,sqrt(length(f)))
    if (abs(real(z))≤1.+tol) && (abs(imag(z))≤tol)
       cfs=cauchylegendreforward(length(f),z)
       dotu(cfs,f.coefficients)
    else
       cfs=cauchylegendrebackward(z)
       m=min(length(f),length(cfs))
       dotu(cfs[1:m],f.coefficients[1:m])
    end
end

function cauchy(s::Bool,f::Fun{Jacobi},z::Number)
    @assert space(f).a==0 && space(f).b==0
    @assert domain(f)==Interval()

   cfs=cauchylegendreforward(s,length(f),z)
   dotu(cfs,f.coefficients)
end


# Sum over all inverses of fromcanonical, see [Olver,2014]
function cauchy{S,L<:Line,T}(f::Fun{MappedSpace{S,L,T}},z)
    @assert domain(f)==Line()
    p=Fun(f.coefficients,space(f).space)
    cauchy(p,tocanonical(f,z)) + cauchy(p,(-1-sqrt(1+4z.^2))./(2z))
end
cauchy(f::Fun{Chebyshev},z)=cauchy(Fun(f,Legendre(domain(f))),z)
cauchy(s,f::Fun{Chebyshev},z)=cauchy(s,Fun(f,Legendre(domain(f))),z)

