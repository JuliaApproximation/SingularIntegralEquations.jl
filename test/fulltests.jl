using ApproxFun, SingularIntegralEquations, Compat.Test
    import ApproxFun: testbandedoperator, testraggedbelowoperator, testblockbandedoperator

include("runtests.jl")






println("Full tests")
## Memory CurveTests

d=exp(im*Interval(0.1,0.2))
x=Fun(d)
w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))
testbandedoperator(Hilbert(space(w)))


## 3 domain ideal fluid flow

Γ=Segment(-im,1.0-im) ∪ Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6))) ∪ Circle(2.0,0.2)
    z=Fun(Γ)

S=PiecewiseSpace(map(d->isa(d,Circle) ? Fourier(d) : JacobiWeight(0.5,0.5,Ultraspherical(1,d)),components(Γ)))
H=Hilbert(S)
testblockbandedoperator(H)

#  TODO: fix testraggedbelowoperator(H)

Ai = [Operator(Fun(ones(component(Γ,1)),Γ)) Fun(ones(component(Γ,2)),Γ) Fun(ones(component(Γ,3)),Γ) real(H)]

@test ApproxFun.israggedbelow(Ai)
@test ApproxFun.israggedbelow(Ai.ops[4])
@test ApproxFun.israggedbelow(Ai.ops[4].op)



B=ApproxFun.SpaceOperator(ApproxFun.BasisFunctional(3),S,ApproxFun.ConstantSpace(Float64))

Ai=[Operator(0)                 0                 0                 B;
    Fun(ones(component(Γ,1)),Γ) Fun(ones(component(Γ,2)),Γ) Fun(ones(component(Γ,3)),Γ) real(H)]



@time testblockbandedoperator(Ai)


k=114;
    α=exp(k/50*im)
    @time a,b,c,ui= Ai \ [0; imag(α*z)]

eltype(Ai)
real(H) |>eltype
components(real(H)*ui)[1] - components(imag(α*z))[1] |> norm
B*ui |> Number


[a;b;c;ui].coefficients
Matrix(Ai[1:400,1:400]) * pad([a;b;c;ui].coefficients,400) - Fun([0; imag(α*z)], rangespace(Ai)).coefficients

qrfact(Matrix(Ai[1:400,1:400]))
a,b,c,ui=Fun(domainspace(Ai), svdfact(Matrix(Ai[1:400,1:400]))  \ pad(Fun([0; imag(α*z)], rangespace(Ai)).coefficients,400))
a|>Number
b|>Number
c|>Number
A = Matrix(Ai[1:400,1:400])
U, σ, V = svd(A)


QR = qrfact(Ai)
    ApproxFun.resizedata!(QR, :, 500)
    @time a,b,c,ui = QR \ [0;imag(α*z)]


QR.R
n = 400
Q = zeros(n,n)
    for j = 1:n
        Q[j, :] =  pad(ApproxFun.Ac_mul_B_coefficients(QR[:Q], [zeros(j-1);1.0]), n)
    end



norm(Q*QR.R.data[1:400,1:400] - Ai[1:400,1:400])

U*diagm(Σ)*V' - A

Q,R = qr(Ai[1:400,1:400])
b = pad(Fun([0; imag(α*z)], rangespace(Ai)).coefficients,400)
R[1:end-1, 1:end-1] \ (Q'*b)[1:end-1]

b = Fun([0; imag(α*z)], rangespace(Ai))

(QR[:Q]'*b ).coefficients
QR.R[1:400,1:400] \pad((QR[:Q]'*b ).coefficients,400) - ui_c

y = pad((QR[:Q]'*b ).coefficients,400)


view(QR.R.data, 1:400, 1:400) \ y

typeof(b)


rangespace(Ai)[2]
space(imag(α*z))
z|>space
[0; imag(α*z)].coefficients
v = imag(α*z).coefficients
@which ApproxFun.sumspacecoefficients(v, space(imag(α*z)), rangespace(Ai)[2])
Fun([0; imag(α*z)], rangespace(Ai)).coefficients |> norm
b = pad(Fun([0; imag(α*z)], rangespace(Ai)).coefficients,400)
σ
ui_c = V[:,1:end-1]*(σ[1:end-1] .\ (U'*b)[1:end-1])


a,b,c,ui = Fun(domainspace(Ai), ui_c)

u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)



@test u(1.1,0.2) ≈ (-0.8290718508107162+0.511097153754im)

println("Example Tests")
include("ExamplesTest.jl")


println("WienerHopfTest")
include("WienerHopfTest.jl")


println("Ideal Fluid Flow tests")
include("IdealFluidFlowTest.jl")
