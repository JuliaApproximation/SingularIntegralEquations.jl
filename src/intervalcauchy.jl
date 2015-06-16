import ApproxFun: dotu,SliceOperator


# This solves as a boundary value provblem

jacobiop(S::PolynomialSpace)=Recurrence(S).'  #.'
jacobiop(S::JacobiWeight)=jacobiop(S.space)

function cauchybackward(S::FunctionSpace,z::Number)
    J=SliceOperator(jacobiop(S)-z,1,0,1)  # drop first row
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

cauchyforward(sp::FunctionSpace,n,z)=forwardsubstitution(jacobiop(sp)-z,n,
                        cauchymoment(sp,1,z),cauchymoment(sp,2,z))

cauchyforward(s::Bool,sp::FunctionSpace,n,z)=forwardsubstitution(jacobiop(sp)-z,n,
                        cauchymoment(s,sp,1,z),cauchymoment(s,sp,2,z))


function cauchyintervalrecurrence(f::Fun,z)
    tol=1./ifloor(Int,sqrt(length(f)))
    if (abs(real(z))≤1.+tol) && (abs(imag(z))≤tol)
       cfs=cauchyforward(space(f),length(f),z)
       dotu(cfs,coefficients(f))
    else
       cfs=cauchybackward(space(f),z)
       m=min(length(f),length(cfs))
       dotu(cfs[1:m],coefficients(f)[1:m])
    end
end


function cauchy{PS<:PolynomialSpace}(f::Fun{PS},z::Number)
    if domain(f)==Interval()
        #TODO: check tolerance
        cauchyintervalrecurrence(Fun(f,Legendre()),z)
    else
        @assert isa(domain(f),Interval)
        cauchy(setdomain(f,Interval()),tocanonical(f,z))
    end
end

function cauchy{PS<:PolynomialSpace}(s::Bool,f::Fun{PS},z::Number)
    @assert domain(f)==Interval()

   cfs=cauchyforward(s,Legendre(),length(f),z)
   dotu(cfs,coefficients(f,Legendre()))
end


# Sum over all inverses of fromcanonical, see [Olver,2014]
function cauchy{S,L<:Line,T}(f::Fun{MappedSpace{S,L,T}},z)
    @assert domain(f)==Line()
    p=Fun(f.coefficients,space(f).space)
    cauchy(p,tocanonical(f,z)) + cauchy(p,(-1-sqrt(1+4z.^2))./(2z))
end



