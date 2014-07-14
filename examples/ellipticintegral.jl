using ApproxFun, RiemannHilbert

function ellipticintegral(a)
    xp=Fun(identity,[a,1.])
    xm=Fun(identity,[-a,-1.])
    up=2.im./(sqrt(1+xp).*sqrt(a+xp).*sqrt(xp-1.).*sqrt(a-xp))
    um=2.im./(sqrt(-1-xm).*sqrt(a+xm).*sqrt(1.-xm).*sqrt(a-xm))
    g(z)=cauchyintegral(up,z)+cauchyintegral(um,z)
    g
end

