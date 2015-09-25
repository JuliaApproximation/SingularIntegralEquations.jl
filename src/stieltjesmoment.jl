##
# cauchymoment implements the moments of the cauchy transform of a space
#  note that it starts from k==1
##


cauchymoment(S...)=stieltjesmoment(S...)/(-2π*im)



# gives \int_-1^1 x^(k-1)/(z-x) dx
function stieltjeslegendremoment(k::Integer,z)
    if k==1
        return -(log(z-1)-log(z+1))
    elseif k==2
        return -(2-z*log(1+z)+z*log(z-1))
    end

    error("stieltjesmoment not implemented for "*string(S.a)*string(S.b))
end

function stieltjeslegendremoment(s::Bool,k::Integer,z)
    if k==1
        return -(log(1-z)+(s?1:-1)*π*im-log(z+1))
    elseif k==2
        return -(2-z*log(1+z)+z*log(1-z) + (s?1:-1)*π*im*z)
    end

    error("stieltjesmoment not implemented for "*string(S.a)*string(S.b))
end


stieltjesmoment(S::Chebyshev,k::Integer,z)=stieltjeslegendremoment(k,tocanonical(S,z))
function stieltjesmoment(J::Jacobi,k::Integer,z)
    if k==1
        return stieltjeslegendremoment(k,z)
    elseif k==2
        if J.a==J.b
            return (1+J.a)*stieltjeslegendremoment(k,z)
        else
            return (J.b-J.a)/2*stieltjeslegendremoment(1,z) + (2+J.a+J.b)/2*stieltjeslegendremoment(2,z)
        end
    end

    error("Not implemented")
end

function stieltjesmoment(s::Bool,S::PolynomialSpace,k::Integer,z)
    z=tocanonical(S,z)

    if k==1
        return -(log(1-z)+(s?1:-1)*π*im-log(z+1))
    elseif k==2
        return -(2-z*log(1+z)+z*log(1-z) + (s?1:-1)*π*im*z)
    end

    error("stieltjesmoment not implemented for "*string(S.a)*string(S.b))
end


# represents atan(sqrt(z))/sqrt(z)
function sqrtatansqrt(x)
    if  isreal(x) && x ≤ 0
        y=sqrt(-x)
        log((1+y)/(1-y))/(2y)
    else
        sqrt(1/x)*atan(sqrt(x))
    end
end

function sqrtatansqrt(s::Bool,x)
    y=sqrt(-x)
    (log((1+y)/(y-1))-(s?1:-1)*π*im)/(2y)
end


#TODO: this is for x^k but we probably want P_k

#These formulae are from mathematica
function stieltjesjacobimoment(α,β,k::Integer,z)
    if α == β == 0
        return stieltjeslegendremoment(k,z)
    elseif isapprox(α,0) && isapprox(β,0.5)
        if k==1
            return -2*sqrt(2)*(sqrtatansqrt(2/(z-1))-1)
        elseif k==2
            return 2*sqrt(2)*(1/3*(3z-2)-z*sqrtatansqrt(2/(z-1)))
        end
    elseif isapprox(α,0) && isapprox(β,-0.5)
        if k==1
            return 2sqrt(1/(z-1))*atan(sqrt(2/(z-1)))
        elseif k==2
            return z/sqrt(1-z)*log((1+1/sqrt(1-z))*(1-sqrt(2)/sqrt(1-z))/((1-1/sqrt(1-z))*(1+sqrt(2)/sqrt(1-z))))-
                2*sqrt(2)-2z/sqrt(1-z)*asinh(sqrt(-1/z))
        end
    elseif !isapprox(α,0) && isapprox(β,0.)
        return (-1)^k*stieltjesjacobimoment(β,α,k,-z)
    end
    error("stieltjesmoment not implemented for JacobiWeight "*string(α)*string(β))
end



function stieltjesjacobimoment(s::Bool,α,β,k::Integer,z)
    if α == β == 0
        return stieltjeslegendremoment(s,k,z)
    elseif isapprox(α,0) && isapprox(β,0.5)
        if k==1
            return -2*sqrt(2)*(sqrtatansqrt(!s,2/(z-1))-1)
        elseif k==2
            return 2*sqrt(2)*(1/3*(3z-2)-z*sqrtatansqrt(!s,2/(z-1)))
        end
    elseif isapprox(α,0) && isapprox(β,-0.5)
        if k==1
            return 2im*(s?-1:1)*sqrt(1/(1-z))*atan((s?-1:1)*im*sqrt(2/(1-z)))
        elseif k==2 && real(z)<0
            return z/sqrt(1-z)*(logabs((1+1/sqrt(1-z))*(1-sqrt(2)/sqrt(1-z))/((1-1/sqrt(1-z))*(1+sqrt(2)/sqrt(1-z))))-π*im*(s?1:-1))-
                            2*sqrt(2)-2z/sqrt(1-z)*asinh(sqrt(-1/z))
        elseif k==2
            return z/sqrt(1-z)*log((1+1/sqrt(1-z))*(1-sqrt(2)/sqrt(1-z))/((1-1/sqrt(1-z))*(1+sqrt(2)/sqrt(1-z))))-
                             2*sqrt(2)-2z/sqrt(1-z)*asinh(im*(s?1:-1)*sqrt(1/z))
        end
    elseif !isapprox(α,0) && isapprox(β,0.)
        return (-1)^k*stieltjesjacobimoment(!s,β,α,k,-z)
    end
    error("stieltjesmoment not implemented for JacobiWeight "*string(α)*string(β))
end



function stieltjesmoment{DD}(S::JacobiWeight{Jacobi{DD},DD},k::Integer,z)
    z=tocanonical(S,z)
    J=S.space

    if k==1
        return stieltjesjacobimoment(S.α,S.β,k,z)
    elseif k==2
        if J.a==J.b
            return (1+J.a)*stieltjesjacobimoment(S.α,S.β,k,z)
        else
            return (J.a-J.b)/2*stieltjesjacobimoment(S.α,S.β,1,z) + (2+J.a+J.b)/2*stieltjesjacobimoment(S.α,S.β,2,z)
        end
    end
    error("stieltjesmoment not implemented for JacobiWeight "*string(S.α)*string(S.β))
end


stieltjesmoment{DD}(S::JacobiWeight{Chebyshev{DD},DD},k::Integer,z)=stieltjesjacobimoment(S.α,S.β,k,tocanonical(S,z))


function stieltjesmoment{DD}(s::Bool,S::JacobiWeight{Jacobi{DD},DD},k::Integer,z)
    z=tocanonical(S,z)
    J=S.space

    if k==1
        return stieltjesjacobimoment(s,S.α,S.β,k,z)
    elseif k==2
        if J.a==J.b
            return (1+J.a)*stieltjesjacobimoment(s,S.α,S.β,k,z)
        else
            return (J.a-J.b)/2*stieltjesjacobimoment(s,S.α,S.β,1,z) + (2+J.a+J.b)/2*stieltjesjacobimoment(s,S.α,S.β,2,z)
        end
    end
    error("stieltjesmoment not implemented for JacobiWeight "*string(S.α)*string(S.β))
end


stieltjesmoment{DD}(s::Bool,S::JacobiWeight{Chebyshev{DD},DD},k::Integer,z)=stieltjesjacobimoment(s,S.α,S.β,k,tocanonical(S,z))




function hilbertmoment(S::JacobiWeight,k::Integer,x)
    x=tocanonical(S,x)

    if S.α == 0 && S.β == 0.5
        if k==1
            if isapprox(x,1.0)
                return 2*sqrt(2)/π
            else
                y=sqrt(-2/(x-1))
                return (2*sqrt(2)-sqrt(2)*(log((1+y)/(y-1))/y))/π
            end
        elseif k==2

        end
    elseif S.α == 0.5 && S.β == 0.

    end
    (stieltjesmoment(true,S,k,x)+stieltjesmoment(false,S,k,x))/(2π)::Float64
end
