#
# hierarchicalsolve
#

# Continuous analogues for low-rank operators case

\{U<:Operator,V<:LowRankOperator}(H::HierarchicalMatrix{U,V},f::Fun) = hierarchicalsolve(H,f)
\{U<:Operator,V<:LowRankOperator,F<:Fun}(H::HierarchicalMatrix{U,V},f::Vector{F}) = hierarchicalsolve(H,f)

hierarchicalsolve{U<:Operator,V<:LowRankOperator}(H::HierarchicalMatrix{U,V},f::Fun) = hierarchicalsolve(H,[f])[1]

hierarchicalsolve(H::Operator,f::Fun) = H\f
hierarchicalsolve{F<:Fun}(H::Operator,f::Vector{F}) = vec(H\transpose(f))

function hierarchicalsolve{U<:Operator,V<:LowRankOperator,F<:Fun}(H::HierarchicalMatrix{U,V},f::Vector{F})
    N,nf = length(space(first(f))),length(f)

    # Pre-compute Factorization

    !isfactored(H) && factorize!(H)

    # Partition HierarchicalMatrix

    (H11,H22),(H21,H12) = partitionmatrix(H)

    # Off-diagonal low-rank matrix assembly

    U12,V12 = strippiecewisespace(H12.U),strippiecewisespace(H12.V)
    U21,V21 = strippiecewisespace(H21.U),strippiecewisespace(H21.V)

    # Partition the right-hand side

    (f1,f2) = partitionfun(f)

    # Solve recursively

    H22f2,H11f1 = hierarchicalsolve(H22,f2),hierarchicalsolve(H11,f1)

    # Compute pivots

    v12,v21 = computepivots(V12,V21,H11f1,H22f2,H.factorization,nf)

    # Solve again with updated right-hand sides

    RHS1 = f1 - At_mul_B(v12,U12)
    RHS2 = f2 - At_mul_B(v21,U21)

    sol = [hierarchicalsolve(H11,RHS1);hierarchicalsolve(H22,RHS2)]
    if N == 2
        return collect(mapreduce(i->depiece(sol[i:nf:end]),vcat,1:nf))
    else
        ls = length(sol)
        return collect(mapreduce(i->depiece(mapreduce(k->pieces(sol[k]),vcat,i:nf:ls)),vcat,1:nf))
    end
end

function factorize!{U<:Operator,V<:LowRankOperator}(H::HierarchicalMatrix{U,V})
    # Partition HierarchicalMatrix

    (H11,H22),(H21,H12) = partitionmatrix(H)

    # Off-diagonal low-rank matrix assembly

    U12,V12 = strippiecewisespace(H12.U),strippiecewisespace(H12.V)
    U21,V21 = strippiecewisespace(H21.U),strippiecewisespace(H21.V)

    # Solve recursively

    H22U21,H11U12 = hierarchicalsolve(H22,U21),hierarchicalsolve(H11,U12)

    # Compute A

    r1,r2 = length(V12),length(V21)
    for i=1:r1,j=1:r2
        H.A[i,j+r1] += V12[i]*H22U21[j]
        H.A[j+r2,i] += V21[j]*H11U12[i]
    end

    # Compute factorization

    H.factorization = pivotldufact(H.A,r1,r2)#lufact(H.A)
    H.factored = true
end

function computepivots{A1,A2,T}(V12::Vector{Functional{T}},V21::Vector{Functional{T}},H11f1::Vector{Fun{A1,T}},H22f2::Vector{Fun{A2,T}},A::PivotLDU{T},nf::Int)
    r1,r2 = length(V12),length(V21)
    b1,b2 = zeros(T,r1,nf),zeros(T,r2,nf)
    for i=1:nf
        for j=1:r1
            b1[j,i] += V12[j]*H22f2[i]
        end
        for j=1:r2
            b2[j,i] += V21[j]*H11f1[i]
        end
    end
    A_ldiv_B1B2!(A,b1,b2)
end






# Utilities

# strippiecewisespace is used for the 2 x 2 case, since the off-diagonal blocks had
# to be in a PiecewiseSpace to conform with higher levels in the hierarchy, yet they only have one space.

strippiecewisespace(v)=v
strippiecewisespace{PWS<:PiecewiseSpace,T}(U::Fun{PWS,T}) = length(space(U)) == 1 ? pieces(U) : U
strippiecewisespace{PWS<:PiecewiseSpace,T}(U::Vector{Fun{PWS,T}}) = length(space(first(U))) == 1 ? mapreduce(pieces,vcat,U) : U

function partitionfun{PWS<:PiecewiseSpace,T}(f::Fun{PWS,T})
    N = length(space(f))
    N2 = div(N,2)
    if N2 == 1
        return (pieces(f)[1]),(pieces(f)[2])
    else
        return (depiece(pieces(f)[1:N2])),(depiece(pieces(f)[1+N2:N]))
    end
end

function partitionfun{PWS<:PiecewiseSpace,T}(f::Vector{Fun{PWS,T}})
    N = length(space(first(f)))
    N2 = div(N,2)
    if N2 == 1
        return (map(x->pieces(x)[1],f),map(x->pieces(x)[2],f))
    else
        return (map(x->depiece(pieces(x)[1:N2]),f),map(x->depiece(pieces(x)[1+N2:N]),f))
    end
end
