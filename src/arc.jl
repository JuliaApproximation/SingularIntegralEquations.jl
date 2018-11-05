

## Cauchy

# pseudocauchy does not normalize at ∞
pseudostieltjes(S::Space{<:Arc},f,z) = stieltjes(setcanonicaldomain(S),f,mobius(S,z))
pseudohilbert(S::Space{<:Arc},f,z) = hilbert(setdomain(S,ChebyshevInterval()),f,mobius(S,z))


stieltjes(S::Space{<:Arc},f,z) =
    stieltjes(setcanonicaldomain(S),f,mobius(S,z))-stieltjes(setcanonicaldomain(S),f,mobius(S,Inf))
function hilbert(S::Space{<:Arc},f,z)
    y = mobius(S,z)
    abs(imag(y)) < 10E-15 || throw(ArgumentError())
    hilbert(setcanonicaldomain(S),f,real(y))+(1/π)*stieltjes(setcanonicaldomain(S),f,mobius(S,Inf))
end



function PseudoHilbert(sp::JacobiWeight{LS,RR},k::Integer) where {LS,RR<:Arc}
    @assert k==1
    csp=setcanonicaldomain(sp)
    H=Hilbert(csp)
    PseudoHilbertWrapper(SpaceOperator(H,setdomain(domainspace(H),domain(sp)),setdomain(rangespace(H),domain(sp))),k)
end


function Hilbert(sp::JacobiWeight{LS,RR},k::Integer) where {LS,RR<:Arc}
    @assert k==1
    csp=setcanonicaldomain(sp)
    St = Stieltjes(csp,mobius(sp,Inf))
    H=Hilbert(csp)+(1/π)*SpaceOperator(St,csp,ConstantSpace(eltype(St)))
    HilbertWrapper(SpaceOperator(H,setdomain(domainspace(H),domain(sp)),setdomain(rangespace(H),domain(sp))),k)
end



## stieltjesintegral

function stieltjesintegral(sp::Space{<:Arc},w,z)
    g=Fun(setcanonicaldomain(sp),w)*fromcanonicalD(sp)
    stieltjesintegral(g,mobius(sp,z))-
        stieltjesintegral(g,mobius(sp,Inf))+
        sum(Fun(sp,w))*log(z-fromcanonical(sp,Inf))
end


function linesumstieltjesintegral(sp::Space{<:Arc},w,z)
    g=Fun(setcanonicaldomain(sp),w)*abs(fromcanonicalD(sp))
    stieltjesintegral(g,mobius(sp,z))-
        stieltjesintegral(g,mobius(sp,Inf))+
        linesum(Fun(sp,w))*log(z-fromcanonical(sp,Inf))
end


function logkernel(sp::Space{<:Arc},w,z)
    g=Fun(setcanonicaldomain(sp),w)*abs(fromcanonicalD(sp))
    logkernel(g,mobius(sp,z))-
        logkernel(g,mobius(sp,Inf))+
        linesum(Fun(sp,w))*logabs(z-fromcanonical(sp,Inf))/π
end


function SingularIntegral(S::JacobiWeight{JW,RR},k::Integer) where {JW,RR<:Arc}
    d=domain(S)
    if k==0
        tol=1E-15
        # the mapped logkernel
        csp=setcanonicaldomain(S)
        Σ=SingularIntegral(csp,0)
        M=Multiplication(abs(fromcanonicalD(d,Fun(identity,csp))),csp)

        z∞=mobius(d,Inf)
        cnst=Vector{Float64}()
        for j = 1:10000
            push!(cnst,logkernel(Fun(csp,[zeros(j-1);1.]),z∞))
            if j ≥ 3 && norm(cnst[end-2:end]) < tol
                break
            end
        end
        L∞=FiniteOperator(cnst',csp,ConstantSpace(eltype(cnst)))

        x=Fun(identity,S)
        SpaceOperator((Σ-L∞)*M,S,setdomain(rangespace(Σ),d)) +
            (logabs(x-fromcanonical(d,Inf))/π)*DefiniteLineIntegral(S)
    else
        # multiply by abs(M')/M' to change to dz to ds
        Mp=fromcanonicalD(d)
        Hilbert(S,k)[setdomain(abs(Mp)/Mp,d)]
    end
end
