export clustertree

#
# clustertree partitions a list of objects x by minimizing distances
# along coordinate axes in 2D.
# It uses complex numbers to represent points in the plane R², returning
# a HierarchicalVector of integers.
#

clustertree(d::UnionDomain) = clustertree(d.domains)

clustertree(x::Tuple) = clustertree(collect(x))

function clustertree(x::AbstractVector)
    A = adjacency(x)
    c = map(center,x)
    clustertree(A,c)
end

clustertree(A::AbstractMatrix,c::AbstractVector) = clustertree(A,c,1:size(A,1))

function clustertree(A::AbstractMatrix,c::AbstractVector,p::AbstractVector)
    i,j = myindmax(A,p)

    cmax = c[i]-c[j]
    rbar,ibar = abs2(real(cmax)),abs2(imag(cmax))
    rmid,imid = reim(.5(c[i]+c[j]))

    if rbar > ibar
        np₁,np₂ = 0,0
        for k in p
            if real(c[k]) < rmid
                np₁+=1
            else
                np₂+=1
            end
        end
        p₁,p₂ = zeros(Int,np₁),zeros(Int,np₂)
        kp₁,kp₂ = 1,1
        for k in p
            if real(c[k]) < rmid
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
            return HierarchicalVector((clustertree(A,c,p₁),clustertree(A,c,p₂)))
        end
    else
        np₁,np₂ = 0,0
        for k in p
            if imag(c[k]) < imid
                np₁+=1
            else
                np₂+=1
            end
        end
        p₁,p₂ = zeros(Int,np₁),zeros(Int,np₂)
        kp₁,kp₂ = 1,1
        for k in p
            if imag(c[k]) < imid
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
            return HierarchicalVector((clustertree(A,c,p₁),clustertree(A,c,p₂)))
        end
    end
end

function adjacency(x::AbstractVector)
    n = length(x)
    A = zeros(real(mapreduce(eltype,promote_type,x)),n,n)
    for j=1:n,i=1:j
        @inbounds A[i,j] = dist2(x[i],x[j])
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
