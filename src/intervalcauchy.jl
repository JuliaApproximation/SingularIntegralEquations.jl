import ApproxFun: dotu,SliceOperator


# This solves as a boundary value provblem
function cauchylegendrebackward(z::Number)
    S=Legendre()
    J=SliceOperator(Recurrence(S).'-z,1,0,1)  # drop first row
    [BasisFunctional(1),
        J]\[cauchymoment(S,1,z)]
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
                        cauchymoment(Legendre(),1,z),cauchymoment(Legendre(),2,z))

cauchylegendreforward(s::Bool,n,z)=forwardsubstitution(Recurrence(Jacobi(0.,0.)).'-z,n,
                        cauchymoment(s,Legendre(),1,z),cauchymoment(s,Legendre(),2,z))

#.'
function cauchy(f::Fun{Jacobi},z::Number)
    if domain(f)==Interval()
        @assert space(f).a==0 && space(f).b==0
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
    else
        @assert isa(domain(f),Interval)
        cauchy(setdomain(f,Interval()),tocanonical(f,z))
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

