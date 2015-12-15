import DualNumbers

export ∇n

function gradient(f::Function,Γ::Domain)
    fx = Fun(x->f(DualNumbers.Dual(real(x),1),imag(x)),Γ)
    fy = Fun(x->f(real(x),DualNumbers.Dual(imag(x),1)),Γ)
    Fun(DualNumbers.epsilon(fx.coefficients),Γ),Fun(DualNumbers.epsilon(fy.coefficients),Γ)
end

∇n(f::Function,d::Domain) = normalderivative(f,d)

normalderivative(f::Function,Γ::UnionDomain) = depiece(map(d->normalderivative(f,d),Γ))

function normalderivative(f::Function,Γ::Domain)
    fx,fy = gradient(f,Γ)
    rp = Fun(x->fromcanonical(Γ,x),Γ)'
    n = -im*rp/abs(rp)
    fx*real(n)+fy*imag(n)
end
