

ApproxFun.opcount(::Union(Hilbert,Cauchy))=1
ApproxFun.opcount(M::HilbertWrapper)=1+ApproxFun.opcount(M.op)


ApproxFun.add_edges!(D::Hilbert,nd,M,labels)=(labels[nd]=string(nd)*":"*(D.order==1?"\$H":"\$H\^"*string(D.order))*"\$")
ApproxFun.add_edges!(D::Cauchy,nd,M,labels)=(labels[nd]=string(nd)*":\$C\$")


for (WRAP,STR) in ((:HilbertWrapper,:"Hw"),)
    @eval ApproxFun.add_edges!(A::$WRAP,nd,M,labels)=ApproxFun.treeadd_edges!(string(nd)*":"*$STR,[A.op],nd,M,labels)
end