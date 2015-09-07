#
# hierarchicalsolve
#

# Continuous analogues for low-rank operators case

\{U<:Operator,V<:LowRankOperator}(H::HierarchicalMatrix{U,V},f::Fun) = hierarchicalsolve(H,f)
\{U<:Operator,V<:LowRankOperator,F<:Fun}(H::HierarchicalMatrix{U,V},f::Vector{F}) = hierarchicalsolve(H,f)

hierarchicalsolve{U<:Operator,V<:LowRankOperator}(H::HierarchicalMatrix{U,V},f::Fun) = hierarchicalsolve(H,[f])[1]

hierarchicalsolve{V<:Operator}(H::V,f::Fun) = H\f
hierarchicalsolve{V<:Operator,F<:Fun}(H::V,f::Vector{F}) = vec(H\transpose(f))

function hierarchicalsolve{U<:Operator,V<:LowRankOperator,F<:Fun}(H::HierarchicalMatrix{U,V},f::Vector{F})
    N,nf = length(space(first(f))),length(f)

    # Pre-compute Factorization

    if !isfactored(H) factorize!(H) end

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

    v12,v21 = computepivots(V12,V21,H11f1,H22f2,H,nf)

    # Solve again with updated right-hand sides

    RHS1 = f1 - At_mul_B(v12,U12)
    RHS2 = f2 - At_mul_B(v21,U21)

    sol = [hierarchicalsolve(H11,RHS1),hierarchicalsolve(H22,RHS2)]
    if N == 2
        return [mapreduce(i->depiece(sol[i:nf:end]),vcat,1:nf)]
    else
        ls = length(sol)
        return [mapreduce(i->depiece(mapreduce(k->pieces(sol[k]),vcat,i:nf:ls)),vcat,1:nf)]
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
        H.A[i,j+r1] += dot(V12[i],H22U21[j])
        H.A[j+r2,i] += dot(V21[j],H11U12[i])
    end

    # Compute factorization

    H.factorization = lufact(H.A)
    H.factored = true
end

function computepivots{VS1,VT1,VS2,VT2,AS1,AT1,AS2,AT2}(V12::Vector{Fun{VS1,VT1}},V21::Vector{Fun{VS2,VT2}},H11f1::Vector{Fun{AS1,AT1}},H22f2::Vector{Fun{AS2,AT2}},H::HierarchicalMatrix,nf::Int)
    T = promote_type(VT1,VT2,AT1,AT2)
    r1,r2 = length(V12),length(V21)
    b = zeros(T,r1+r2,nf)
    for i=1:nf
        for j=1:r1
            b[j,i] = dot(V12[j],H22f2[i])
        end
        for j=1:r2
            b[j+r1,i] = dot(V21[j],H11f1[i])
        end
    end
    vc = H.factorization\b
    vc[1:r1,1:nf],vc[r1+1:r1+r2,1:nf]
end

function computepivots{V1,V2,A1,A2,S,T,P}(V12::Vector{Fun{V1,P}},V21::Vector{Fun{V2,P}},H11f1::Vector{Fun{A1,P}},H22f2::Vector{Fun{A2,P}},H::HierarchicalMatrix{S,T,P},nf::Int)
    r1,r2 = length(V12),length(V21)
    b = zeros(P,r1+r2,nf)
    for i=1:nf
        for j=1:r1
            b[j,i] = dot(V12[j],H22f2[i])
        end
        for j=1:r2
            b[j+r1,i] = dot(V21[j],H11f1[i])
        end
    end
    A_ldiv_B!(H.factorization,b)
    b[1:r1,1:nf],b[r1+1:r1+r2,1:nf]
end






# Utilities

# strippiecewisespace is used for the 2 x 2 case, since the off-diagonal blocks had
# to be in a PiecewiseSpace to conform with higher levels in the hierarchy, yet they only have one space.

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
