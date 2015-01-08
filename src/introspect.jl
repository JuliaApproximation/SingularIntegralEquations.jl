

ApproxFun.treecount(::Union(Hilbert,Cauchy))=1
ApproxFun.treecount(M::HilbertWrapper)=1+ApproxFun.opcount(M.op)


ApproxFun.texname(D::Hilbert)=(labels[nd]=string(nd)*":"*(D.order==1?"\${\\cal H}":"\${\\cal H}\^"*string(D.order))*"\$")
ApproxFun.texname(D::Cauchy)=(labels[nd]=string(nd)*":\${\\cal C}\$")
ApproxFun.texname(D::HilbertWrapper)=(D.order==1?"\$({\\cal H}":"\$({\\cal H}\^"*string(D.order))*")\$"



@eval ApproxFun.add_edges!(A::HilbertWrapper,nd,M,labels)=treeadd_edges!(string(nd)*":"*texname(A),[A.op],nd,M,labels)