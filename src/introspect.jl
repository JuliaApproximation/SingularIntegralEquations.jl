

ApproxFun.treecount(::Union(Hilbert,Stieltjes))=1
ApproxFun.treecount(M::HilbertWrapper)=1+ApproxFun.opcount(M.op)


ApproxFun.texname(D::Hilbert)=(labels[nd]=string(nd)*":"*(D.order==1?"\${\\cal H}":"\${\\cal H}\^"*string(D.order))*"\$")
ApproxFun.texname(D::Stieltjes)=(labels[nd]=string(nd)*":\${\\cal S}\$")
ApproxFun.texname(D::HilbertWrapper)=(D.order==1?"\$({\\cal H}":"\$({\\cal H}\^"*string(D.order))*")\$"



@eval ApproxFun.add_edges!(A::HilbertWrapper,nd,M,labels)=treeadd_edges!(string(nd)*":"*texname(A),[A.op],nd,M,labels)