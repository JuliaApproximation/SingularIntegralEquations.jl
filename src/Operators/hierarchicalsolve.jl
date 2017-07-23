#
# hierarchicalsolve
#

# Continuous analogues for low-rank operators case

\{U<:Operator,V<:AbstractLowRankOperator}(H::HierarchicalOperator{U,V},f::Fun) = hierarchicalsolve(H,f)
\{U<:Operator,V<:AbstractLowRankOperator,F<:Fun}(H::HierarchicalOperator{U,V},f::Vector{F}) = hierarchicalsolve(H,f)

hierarchicalsolve{U<:Operator,V<:AbstractLowRankOperator}(H::HierarchicalOperator{U,V},f::Fun) = hierarchicalsolve(H,[f])[1]

hierarchicalsolve(H::Operator,f::Fun) = H\f
hierarchicalsolve{F<:Fun}(H::Operator,f::Vector{F}) = map(g->H\g,f)#vec(H\transpose(f))

function hierarchicalsolve{U<:Operator,V<:AbstractLowRankOperator,F<:Fun}(H::HierarchicalOperator{U,V},f::Vector{F})
    # Pre-compute Factorization

    !isfactored(H) && factorize!(H)

    # Partition HierarchicalOperator

    (H11,H22),(H21,H12) = partition(H)

    # Off-diagonal low-rank matrix assembly

    U12,V12 = H12.U,H12.V
    U21,V21 = H21.U,H21.V

    # Partition the right-hand side

    f1,f2 = partition(f,space(first(U12)),space(first(U21)))

    # Solve recursively

    H22f2,H11f1 = hierarchicalsolve(H22,f2),hierarchicalsolve(H11,f1)

    # Compute pivots

    v12,v21 = computepivots(V12,V21,H11f1,H22f2,H.factorization)

    # Solve again with updated right-hand sides

    RHS1 = f1 - v12.'*U12
    RHS2 = f2 - v21.'*U21

    sol1,sol2 = hierarchicalsolve(H11,RHS1),hierarchicalsolve(H22,RHS2)

    return assemble(sol1,sol2)
end

function factorize!{U<:Operator,V<:AbstractLowRankOperator}(H::HierarchicalOperator{U,V})
    # Partition HierarchicalOperator

    (H11,H22),(H21,H12) = partition(H)

    # Off-diagonal low-rank matrix assembly

    U12,V12 = H12.U,H12.V
    U21,V21 = H21.U,H21.V

    # Solve recursively

    H22U21,H11U12 = hierarchicalsolve(H22,U21),hierarchicalsolve(H11,U12)

    # Compute A

    fillpivotmatrix!(H.A,V12,V21,H22U21,H11U12)

    # Compute factorization

    H.factorization = pivotldufact(H.A,length(V12),length(V21))#lufact(H.A)
    H.factored = true
end

function fillpivotmatrix!{A1,A2,T}(A::Matrix{T},V12::Vector{Operator{T}},V21::Vector{Operator{T}},H22U21::Vector{Fun{A1,T}},H11U12::Vector{Fun{A2,T}})
    r1,r2 = length(V12),length(V21)
    for i=1:r1,j=1:r2
        @inbounds A[i,j+r1] += V12[i]*H22U21[j]
        @inbounds A[j+r2,i] += V21[j]*H11U12[i]
    end
end

function fillpivotmatrix!{V1,V2,A1,A2,T}(A::Matrix{T},V12::Vector{Fun{V1,T}},V21::Vector{Fun{V2,T}},H22U21::Vector{Fun{A1,T}},H11U12::Vector{Fun{A2,T}})
    r1,r2 = length(V12),length(V21)
    for i=1:r1,j=1:r2
        @inbounds A[i,j+r1] += linebilinearform(V12[i],H22U21[j])
        @inbounds A[j+r2,i] += linebilinearform(V21[j],H11U12[i])
    end
end

function computepivots{A1,A2,T}(V12::Vector{Operator{T}},V21::Vector{Operator{T}},H11f1::Vector{Fun{A1,T}},H22f2::Vector{Fun{A2,T}},A::PivotLDU{T})
    @assert length(H11f1) == length(H22f2)
    nf,r1,r2 = length(H11f1),length(V12),length(V21)
    b1,b2 = zeros(T,r1,nf),zeros(T,r2,nf)
    for i=1:nf
        for j=1:r1
            @inbounds b1[j,i] += V12[j]*H22f2[i]
        end
        for j=1:r2
            @inbounds b2[j,i] += V21[j]*H11f1[i]
        end
    end
    A_ldiv_B1B2!(A,b1,b2)
end

function computepivots{V1,V2,A1,A2,T}(V12::Vector{Fun{V1,T}},V21::Vector{Fun{V2,T}},H11f1::Vector{Fun{A1,T}},H22f2::Vector{Fun{A2,T}},A::PivotLDU{T})
    @assert length(H11f1) == length(H22f2)
    nf,r1,r2 = length(H11f1),length(V12),length(V21)
    b1,b2 = zeros(T,r1,nf),zeros(T,r2,nf)
    for i=1:nf
        for j=1:r1
            @inbounds b1[j,i] += linebilinearform(V12[j],H22f2[i])
        end
        for j=1:r2
            @inbounds b2[j,i] += linebilinearform(V21[j],H11f1[i])
        end
    end
    A_ldiv_B1B2!(A,b1,b2)
end




# Utilities

##
# partition(f::Fun,sp1,sp2) uses the spaces sp1 and sp2
# to achieve type-stability by specializing on the four cases:
#
# [x₁ | ⋯ ⋯ xₙ]
# [x₁ ⋯ | ⋯ xₙ]
# [x₁ ⋯ ⋯ | xₙ]
# [x₁ | x₂]
##

function partition{PWS<:PiecewiseSpace,S1<:PiecewiseSpace,S2<:PiecewiseSpace,T}(f::Fun{PWS,T},sp1::S1,sp2::S2)
    p,N1,N2 = components(f),length(sp1),length(sp2)
    return (depiece(p[1:N1])),(depiece(p[1+N1:N1+N2]))
end

function partition{PWS<:PiecewiseSpace,S1<:PiecewiseSpace,S2,T}(f::Fun{PWS,T},sp1::S1,sp2::S2)
    p,N1 = components(f),length(sp1)
    return (depiece(p[1:N1])),(p[1+N1])
end

function partition{PWS<:PiecewiseSpace,S1,S2<:PiecewiseSpace,T}(f::Fun{PWS,T},sp1::S1,sp2::S2)
    p,N2 = components(f),length(sp2)
    return (p[1]),(depiece(p[2:1+N2]))
end

function partition{PWS<:PiecewiseSpace,S1,S2,T}(f::Fun{PWS,T},sp1::S1,sp2::S2)
    p = components(f)
    return (p[1]),(p[2])
end

function partition{PWS<:PiecewiseSpace,S1,S2,T}(f::Vector{Fun{PWS,T}},sp1::S1,sp2::S2)
    p1 = partition(f[1],sp1,sp2)
    ret1,ret2 = Vector{typeof(p1[1])}(length(f)),Vector{typeof(p1[2])}(length(f))
    ret1[1],ret2[1] = p1[1],p1[2]
    for k=2:length(f)
        @inbounds ret1[k],ret2[k] = partition(f[k],sp1,sp2)
    end
    ret1,ret2
end

##
# assemble(sol1::Vector{Fun},sol2::Vector{Fun}) uses the spaces of sol1 and sol2
# to achieve type-stability by specializing on the four cases:
#
# [s₁ | ⋯ ⋯ sₙ]
# [s₁ ⋯ | ⋯ sₙ]
# [s₁ ⋯ ⋯ | sₙ]
# [s₁ | s₂]
##

function assemble{S1,S2,T1,T2}(sol1::Vector{Fun{S1,T1}},sol2::Vector{Fun{S2,T2}})
    @assert length(sol1) == length(sol2)
    p = depiece((sol1[1],sol2[1]))
    ret = Vector{typeof(p)}(length(sol1))
    ret[1] = p
    for k=2:length(sol1)
        @inbounds ret[k] = depiece((sol1[k],sol2[k]))
    end
    ret
end

function assemble{S1<:PiecewiseSpace,S2<:PiecewiseSpace,T1,T2}(sol1::Vector{Fun{S1,T1}},sol2::Vector{Fun{S2,T2}})
    @assert length(sol1) == length(sol2)
    p = depiece(vcat(components(sol1[1]),components(sol2[1])))
    ret = Vector{typeof(p)}(length(sol1))
    ret[1] = p
    for k=2:length(sol1)
        @inbounds ret[k] = depiece(vcat(components(sol1[k]),components(sol2[k])))
    end
    ret
end

function assemble{S1<:PiecewiseSpace,S2,T1,T2}(sol1::Vector{Fun{S1,T1}},sol2::Vector{Fun{S2,T2}})
    @assert length(sol1) == length(sol2)
    p = depiece(vcat(components(sol1[1]),sol2[1]))
    ret = Vector{typeof(p)}(length(sol1))
    ret[1] = p
    for k=2:length(sol1)
        @inbounds ret[k] = depiece(vcat(components(sol1[k]),sol2[k]))
    end
    ret
end

function assemble{S1,S2<:PiecewiseSpace,T1,T2}(sol1::Vector{Fun{S1,T1}},sol2::Vector{Fun{S2,T2}})
    @assert length(sol1) == length(sol2)
    p = depiece(vcat(sol1[1],components(sol2[1])))
    ret = Vector{typeof(p)}(length(sol1))
    ret[1] = p
    for k=2:length(sol1)
        @inbounds ret[k] = depiece(vcat(sol1[k],components(sol2[k])))
    end
    ret
end
