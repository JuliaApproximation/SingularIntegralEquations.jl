
include("runtests.jl")

## Memory CurveTests

d=exp(im*Interval(0.1,0.2))
x=Fun(d)
w=1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))
testbandedoperator(Hilbert(space(w)))


include("ExamplesTest.jl")
