##
# cauchymoment implements the moments of the cauchy transform of a space
#  note that it starts from k==1
##


cauchymoment(S...)=stieltjesmoment(S...)/(-2π*im)

function stieltjesmoment(S::PolynomialSpace,k::Integer,z)
    if k==1
        return -(log(z-1)-log(z+1))
    elseif k==2
        return -(2-z*log(1+z)+z*log(z-1))
    end

    error("stieltjesmoment not implemented for "*string(S.a)*string(S.b))
end

function stieltjesmoment(s::Bool,S::PolynomialSpace,k::Integer,z)
    if k==1
        return -(log(1-z)+(s?1:-1)*π*im-log(z+1))
    elseif k==2
        return -(2-z*log(1+z)+z*log(1-z) + (s?1:-1)*π*im*z)
    end

    error("stieltjesmoment not implemented for "*string(S.a)*string(S.b))
end

# represents sqrt(z)*atan(sqrt(z))
sqrtatansqrt(z)=sqrt(1/z)*atan(sqrt(z))
function sqrtatansqrt(x::Real)
    if  x ≤0
        y=sqrt(-x)
        (log(1+y)-log(1-y))/(2y)
    else
        sqrt(1/x)*atan(sqrt(x))
    end
end


function stieltjesmoment(S::JacobiWeight,k::Integer,z)
    if S.α == S.β == 0
        return stieltjesmoment(S.space,k,z)
    elseif S.α == 0 && S.β == 0.5
        if k==1
            return -2*sqrt(2)*(sqrtatansqrt(2/(z-1))-1)
        elseif k==2
            return 2*sqrt(2)/3*(3z-2)-2*sqrt(1-z)*z*atanh(sqrt(2)/sqrt(1-z))
        end
    elseif S.α == 0.5 && S.β == 0.
        return (-1)^k*stieltjesmoment(JacobiWeight(0.,0.5,S.space),k,-z)
    end
    error("stieltjesmoment not implemented for JacobiWeight "*string(S.α)*string(S.β))
end
