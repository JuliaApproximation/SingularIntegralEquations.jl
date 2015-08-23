#
# woodburysolve
#

# Classical matrix case

woodburysolve{S<:AbstractMatrix,V}(H::S,f::AbstractVecOrMat{V}) = H\f

function woodburysolve{S<:AbstractMatrix,U<:LowRankMatrix,V}(H::HierarchicalMatrix{S,U},f::AbstractVecOrMat{V})
    T,nf = promote_type(eltype(H),V),size(f,2)

    # Partition HierarchicalMatrix

    (H11,H22),(H21,H12) = partitionmatrix(H)

    # Off-diagonal low-rank matrix assembly

    U12,V12 = H12.U,H12.V
    U21,V21 = H21.U,H21.V

    # Partition the right-hand side

    (f1,f2) = (f[1:size(H11,1),:],f[1+size(H11,1):end,:])

    # Solve recursively

    A2U21f2,A1U12f1 = woodburysolve(H22,hcat(U21,f2)),woodburysolve(H11,hcat(U12,f1))

    # Compute pivots

    v12,v21 = computepivots(V12,V21,A1U12f1,A2U21f2,nf)

    # Solve again with updated right-hand sides

    RHS1 = f1 - At_mul_B(v12,U12)
    RHS2 = f2 - At_mul_B(v21,U21)
    reshape([woodburysolve(H11,RHS1);woodburysolve(H22,RHS2)],size(f))
end


# Continuous analogues for low-rank operators case

woodburysolve{V<:Operator}(H::V,f::Fun) = H\f
woodburysolve{V<:Operator,F<:Fun}(H::V,f::Vector{F}) = vec(H\transpose(f))

woodburysolve{U<:Operator,V<:LowRankOperator}(H::HierarchicalMatrix{U,V},f::Fun) = woodburysolve(H,[f])[1]

function woodburysolve{U<:Operator,V<:LowRankOperator,F<:Fun}(H::HierarchicalMatrix{U,V},f::Vector{F})
    N,nf = length(space(first(f))),length(f)

    # Partition HierarchicalMatrix

    (H11,H22),(H21,H12) = partitionmatrix(H)

    # Off-diagonal low-rank matrix assembly

    U12,V12 = strippiecewisespace(H12.U),strippiecewisespace(H12.V)
    U21,V21 = strippiecewisespace(H21.U),strippiecewisespace(H21.V)

    # Partition the right-hand side

    (f1,f2) = partitionfun(f)

    # Solve recursively

    A2U21f2,A1U12f1 = woodburysolve(H22,vcat(U21,f2)),woodburysolve(H11,vcat(U12,f1))

    # Compute pivots

    v12,v21 = computepivots(V12,V21,A1U12f1,A2U21f2,nf)

    # Solve again with updated right-hand sides

    RHS1 = f1 - At_mul_B(v12,U12)
    RHS2 = f2 - At_mul_B(v21,U21)

    sol = [woodburysolve(H11,RHS1),woodburysolve(H22,RHS2)]
    if N == 2
        return [mapreduce(i->depiece(sol[i:nf:end]),vcat,1:nf)]
    else
        ls = length(sol)
        return [mapreduce(i->depiece(mapreduce(k->pieces(sol[k]),vcat,i:nf:ls)),vcat,1:nf)]
    end
end

# strippiecewisespace is used for the 2 x 2 case, since the off-diagonal blocks had
# to be in a PiecewiseSpace to conform with higher levels in the hierarchy, yet they only have one space.

strippiecewisespace{PWS<:PiecewiseSpace,T}(U::Fun{PWS,T}) = length(space(U)) == 1 ? pieces(U) : U
strippiecewisespace{PWS<:PiecewiseSpace,T}(U::Vector{Fun{PWS,T}}) = length(space(first(U))) == 1 ? mapreduce(pieces,vcat,U) : U

function partitionfun{PWS<:PiecewiseSpace,T}(f::Fun{PWS,T})
    N = length(space(f))
    N2 = div(N,2)
    (depiece(pieces(f)[1:N2])),(depiece(pieces(f)[1+N2:N]))
end

function partitionfun{PWS<:PiecewiseSpace,T}(f::Vector{Fun{PWS,T}})
    N = length(space(first(f)))
    N2 = div(N,2)
    (map(x->depiece(pieces(x)[1:N2]),f),map(x->depiece(pieces(x)[1+N2:N]),f))
end

# Pivot computation

function computepivots{V1,V2,A1,A2}(V12::Matrix{V1},V21::Matrix{V2},A1U12f1::Matrix{A1},A2U21f2::Matrix{A2},nf::Int)
    T = promote_type(V1,V2,A1,A2)
    r1,r2 = size(V12,2),size(V21,2)
    A = eye(T,r1+r2)
    for i=1:r1,j=1:r2
        A[i,j+r1] = dot(V12[:,i],A2U21f2[:,j])
        A[j+r2,i] = dot(V21[:,j],A1U12f1[:,i])
    end
    b = zeros(T,r1+r2,nf)
    for i=1:nf
        for j=1:r1
            b[j,i] = dot(V12[:,j],A2U21f2[:,end-nf+i])
        end
        for j=1:r2
            b[j+r1,i] = dot(V21[:,j],A1U12f1[:,end-nf+i])
        end
    end
    vc = A\b
    vc[1:r1,1:nf],vc[r1+1:r1+r2,1:nf]
end

function computepivots{V1,V2,A1,A2}(V12::Vector{V1},V21::Vector{V2},A1U12f1::Vector{A1},A2U21f2::Vector{A2},nf::Int)
    T = promote_type(eltype(first(V12)),eltype(first(V21)),eltype(first(A1U12f1)),eltype(first(A2U21f2)))
    r1,r2 = length(V12),length(V21)
    A = eye(T,r1+r2)
    for i=1:r1,j=1:r2
        A[i,j+r1] = dot(V12[i],A2U21f2[j])
        A[j+r2,i] = dot(V21[j],A1U12f1[i])
    end
    b = zeros(T,r1+r2,nf)
    for i=1:nf
        for j=1:r1
            b[j,i] = dot(V12[j],A2U21f2[end-nf+i])
        end
        for j=1:r2
            b[j+r1,i] = dot(V21[j],A1U12f1[end-nf+i])
        end
    end
    vc = A\b
    vc[1:r1,1:nf],vc[r1+1:r1+r2,1:nf]
end
