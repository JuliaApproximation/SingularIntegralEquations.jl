export clustertree

clustertree(D::UnionDomain) = clustertree(D.domains)

clustertree(x::Tuple) = clustertree([x...])

function clustertree(x::AbstractVector)
    A = adjacency(x)
    clustertree(A)
end

clustertree(A::AbstractMatrix) = clustertree(A,1:size(A,1))

function clustertree(A::AbstractMatrix,p::AbstractVector)
    i,j = myindmax(A,p)

    np₁,np₂ = 0,0
    for k in p
        if A[k,i] < A[k,j]
            np₁+=1
        else
            np₂+=1
        end
    end
    p₁,p₂ = zeros(Int,np₁),zeros(Int,np₂)
    kp₁,kp₂ = 1,1
    for k in p
        if A[k,i] < A[k,j]
            p₁[kp₁] = k
            kp₁+=1
        else
            p₂[kp₂] = k
            kp₂+=1
        end
    end

    if length(p₁) == 1 || length(p₂) == 1
        return HierarchicalVector((p₁,p₂))
    else
        return HierarchicalVector((clustertree(A,p₁),clustertree(A,p₂)))
    end
end

dist(x,y) = norm(x-y)

function adjacency(x::AbstractVector)
    n = length(x)
    A = zeros(real(mapreduce(eltype,promote_type,x)),n,n)
    for j=1:n,i=1:j
        @inbounds A[i,j] = dist(x[i],x[j])
    end
    for j=1:n,i=1:j
        @inbounds A[j,i] = A[i,j]
    end
    A
end

function myfindmax(a,p)
    if isempty(a)
        throw(ArgumentError("collection must be non-empty"))
    end
    m = a[p[1],p[1]]
    mij = p[1],p[1]
    for i in p,j in p
        aij = a[i,j]
        if aij > m || m!=m
            m = aij
            mij = i,j
        end
    end
    return (m, mij)
end
myindmax(a,p) = myfindmax(a,p)[2]
