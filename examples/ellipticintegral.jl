# This file calcules the ellipticintegral
#
#    ∫_0^z dz/(sqrt(1-z^2)*sqrt(a^2-z^2))
#
# by writing the integrand as a Cauchy transform
# cauchyintegral is an integral of the Cauchy transform


using ApproxFun, SingularIntegralEquations

function ellipticintegral(a)
    xp=Fun(identity,a..1)
    xm=Fun(identity,-a .. -1)
    up=2.0im/(sqrt(1+xp)*sqrt(a+xp)*sqrt(xp-1)*sqrt(a-xp))
    um=2.0im/(sqrt(-1-xm)*sqrt(a+xm)*sqrt(1-xm)*sqrt(a-xm))
    z->-cauchyintegral(up,z)-cauchyintegral(um,z)
end


z=Fun(0..0.5)
ellipticintegral(2.0)(0.5)-sum(1/(sqrt(1-z^2)*sqrt(2.0^2-z^2)))
