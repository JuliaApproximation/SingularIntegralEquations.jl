import ApproxFun: dotu,SliceOperator

function cauchylegendre(z::Number)
    J=SliceOperator(JacobiRecurrence(0.,0.)-z,1,0,1)  # drop first row
    [BasisFunctional(1),
        J]\[(log(z-1)-log(z+1))/(2π*im)]
end

function cauchy(f::Fun{Jacobi},z::Number)
    @assert space(f).a==0 && space(f).b==0
   cfs=cauchylegendre(z)
   m=min(length(f),length(cfs))
   dotu(cfs[1:m],f.coefficients[1:m]) 
end

