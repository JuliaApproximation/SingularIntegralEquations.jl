#
# hierarchicalsolve
#

# Classical matrix case

\{S<:AbstractMatrix,U<:LowRankMatrix}(H::HierarchicalMatrix{S,U},f::AbstractVecOrMat) = hierarchicalsolve(H,f)

hierarchicalsolve(H::AbstractMatrix,f::AbstractVecOrMat) = H\f

function hierarchicalsolve{S<:AbstractMatrix,U<:LowRankMatrix}(H::HierarchicalMatrix{S,U},f::AbstractVecOrMat)
    T,nf = promote_type(eltype(H),eltype(f)),size(f,2)

    # Pre-compute Factorization

    !isfactored(H) && factorize!(H)

    # Partition HierarchicalMatrix

    (H11,H22),(H21,H12) = partition(H)

    # Off-diagonal low-rank matrix assembly

    U12,V12 = H12.U,H12.V
    U21,V21 = H21.U,H21.V

    # Partition the right-hand side

    (f1,f2) = (f[1:size(H11,1),:],f[1+size(H11,1):end,:])

    # Solve recursively

    H22f2,H11f1 = hierarchicalsolve(H22,f2),hierarchicalsolve(H11,f1)

    # Compute pivots

    v12,v21 = computepivots(V12,V21,H11f1,H22f2,H.factorization,nf)

    # Solve again with updated right-hand sides

    RHS1 = f1-U12*v12
    RHS2 = f2-U21*v21
    reshape([hierarchicalsolve(H11,RHS1);hierarchicalsolve(H22,RHS2)],size(f))
end

function factorize!{S<:AbstractMatrix,U<:LowRankMatrix}(H::HierarchicalMatrix{S,U})
    # Partition HierarchicalMatrix

    (H11,H22),(H21,H12) = partition(H)

    # Off-diagonal low-rank matrix assembly

    U12,V12 = H12.U,H12.V
    U21,V21 = H21.U,H21.V

    # Solve recursively

    H22U21,H11U12 = hierarchicalsolve(H22,U21),hierarchicalsolve(H11,U12)

    # Compute A

    r1,r2 = size(V12,2),size(V21,2)
    for i=1:r1,j=1:r2
        H.A[i,j+r1] += dot(V12[:,i],H22U21[:,j])
        H.A[j+r2,i] += dot(V21[:,j],H11U12[:,i])
    end

    # Compute factorization

    H.factorization = pivotldufact(H.A,r1,r2)#lufact(H.A)
    H.factored = true
end

function computepivots{V1,V2,A1,A2,T}(V12::Matrix{V1},V21::Matrix{V2},H11f1::Matrix{A1},H22f2::Matrix{A2},A::Factorization{T},nf::Int)
    P = promote_type(V1,V2,A1,A2)
    r1,r2 = size(V12,2),size(V21,2)
    b = zeros(P,r1+r2,nf)
    for i=1:nf
        for j=1:r1
            b[j,i] += dot(V12[:,j],H22f2[:,i])
        end
        for j=1:r2
            b[j+r1,i] += dot(V21[:,j],H11f1[:,i])
        end
    end
    vc = A\b
    vc[1:r1,1:nf],vc[r1+1:r1+r2,1:nf]
end

function computepivots{T}(V12::Matrix{T},V21::Matrix{T},H11f1::Vector{T},H22f2::Vector{T},A::PivotLDU{T},nf::Int)
    r1,r2 = size(V12,2),size(V21,2)
    b1,b2 = zeros(T,r1),zeros(T,r2)
    for j=1:r1
        b1[j] += dot(V12[:,j],H22f2)
    end
    for j=1:r2
        b2[j] += dot(V21[:,j],H11f1)
    end
    A_ldiv_B1B2!(A,b1,b2)
end

function computepivots{T}(V12::Matrix{T},V21::Matrix{T},H11f1::Matrix{T},H22f2::Matrix{T},A::PivotLDU{T},nf::Int)
    r1,r2 = size(V12,2),size(V21,2)
    b1,b2 = zeros(T,r1,nf),zeros(T,r2,nf)
    for i=1:nf
        for j=1:r1
            b1[j,i] += dot(V12[:,j],H22f2[:,i])
        end
        for j=1:r2
            b2[j,i] += dot(V21[:,j],H11f1[:,i])
        end
    end
    A_ldiv_B1B2!(A,b1,b2)
end
