export clustertree

#
# clustertree partitions a collection x by maximizing distances normalized by the
# minimal diameter along coordinate axes in 2D. It uses complex numbers to represent
# points in the plane R². The Admissibility condition for low-rank approximants to
# asymptotically smooth kernels is:
#
# Adm_η(σ×τ) = min{diam(Γ_σ),diam(Γ_τ)} ≤ 2η dist(Γ_σ,Γ_τ),
#
# and clustertree creates partitions that minimize η, the strength of the admissibility.
#

clustertree(x) = x
clustertree(sp::PiecewiseSpace) = Space(clustertree(domain(sp)))
clustertree(d::UnionDomain) = clustertree(d.domains)

clustertree(x::Tuple) = clustertree(collect(x))

function clustertree(x::AbstractVector)
    A = weightedadjacency(x)
    c = map(centroid,x)
    clustertree(A,c,x)
end

clustertree(A::AbstractMatrix,c::AbstractVector,x::AbstractVector) = clustertree(A,c,x,1:size(A,1))

for (TYP,HTYP) in ((:Number,:HierarchicalVector),(:Domain,:HierarchicalDomain))
    @eval begin
        function clustertree(A::AbstractMatrix,c::AbstractVector,x::AbstractVector{T},p::AbstractVector) where T<:$TYP
            i,j = indmaxp(A,p)

            cmax = c[i]-c[j]
            rbar,ibar = abs2(real(cmax)),abs2(imag(cmax))
            rmid,imid = reim((c[i]+c[j])/2)

            (f,mid) = rbar ≥ ibar ? (real,rmid) : (imag,imid)

            np₁,np₂ = 0,0
            for k in p
                f(c[k]) < mid ? np₁+=1 : np₂+=1
            end
            p₁,p₂ = zeros(Int,np₁),zeros(Int,np₂)
            kp₁,kp₂ = 1,1
            for k in p
                f(c[k]) < mid ? (p₁[kp₁] = k; kp₁+=1) : (p₂[kp₂] = k; kp₂+=1)
            end
            np₁ == 0 && (push!(p₁,p₂[1]); shift!(p₂); np₁+=1; np₂-=1)
            np₂ == 0 && (push!(p₂,p₁[1]); shift!(p₁); np₁-=1; np₂+=1)

            if np₁ == 1 && np₂ == 1
                return $HTYP((x[p₁[1]],x[p₂[1]]))
            elseif np₁ == 1
                return $HTYP((x[p₁[1]],clustertree(A,c,x,p₂)))
            elseif np₂ == 1
                return $HTYP((clustertree(A,c,x,p₁),x[p₂[1]]))
            else
                return $HTYP((clustertree(A,c,x,p₁),clustertree(A,c,x,p₂)))
            end
        end
    end
end

#
# The weighted adjacency matrix computes all the possible distances normalized by the
# minimum of the diameters of the sets. This is related to the Admissibility condition.
#

function weightedadjacency(x::AbstractVector)
    n = length(x)
    A = zeros(real(mapreduce(eltype,promote_type,x)),n,n)
    for j=1:n,i=1:j
        @inbounds A[i,j] = dist2(x[i],x[j])/min(diam2(x[i]),diam2(x[j]))
    end
    for j=1:n,i=1:j-1
        @inbounds A[j,i] = A[i,j]
    end
    A
end

# find the maximum of A over the indices p without allocating A[p,p].

function findmaxp(a,p)
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
indmaxp(a,p) = findmaxp(a,p)[2]
