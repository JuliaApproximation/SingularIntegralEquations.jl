#
# woodburysolve
#

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

    r1,r2 = rank(H12),rank(H21)
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
    v12,v21 = vc[1:r1,:],vc[r1+1:end,:]

    # Solve again with updated right-hand sides

    RHS1 = f1-U12*v12
    RHS2 = f2-U21*v21
    reshape([woodburysolve(H11,RHS1);woodburysolve(H22,RHS2)],size(f))
end
