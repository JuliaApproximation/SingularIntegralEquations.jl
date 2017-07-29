import DualNumbers

export ∇n

function gradient(f::Function,Γ::Domain)
    fx = Fun(x->f(DualNumbers.Dual(real(x),1),imag(x)),Γ)
    fy = Fun(x->f(real(x),DualNumbers.Dual(imag(x),1)),Γ)
    Fun(Γ,DualNumbers.epsilon.(fx.coefficients)),Fun(Γ,DualNumbers.epsilon.(fy.coefficients))
end

∇n(f::Function,d::Domain) = normalderivative(f,d)

normalderivative(f::Function,Γ::UnionDomain) = Fun(map(d->normalderivative(f,d),Γ),PiecewiseSpace)

function normalderivative(f::Function,Γ::Domain)
    fx,fy = gradient(f,Γ)
    rp = Fun(x->fromcanonical(Γ,x),Γ)'
    n = -im*rp/abs(rp)
    fx*real(n)+fy*imag(n)
end
