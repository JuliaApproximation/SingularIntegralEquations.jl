using Test, ApproxFun, SingularIntegralEquations

@testset "Nonlocal operators" begin
    α = 2.5
    δ = 0.1
    for sp in (Fourier(), SinSpace(), CosSpace(), Laurent(), Hardy{true}(PeriodicInterval()), Hardy{false}(PeriodicInterval()))
        L = NonlocalLaplacian(sp, α, δ)
        @test domain(L) == domain(sp)
        @test domainspace(L) == rangespace(L) == sp
        @test bandinds(L) == (0,0)
    end

    NonlocalLaplacian(PeriodicInterval(), α, δ) === NonlocalLaplacian(Space(PeriodicInterval()), α, δ)

    L = NonlocalLaplacian(SinSpace(), α, δ)
    @test norm(L[10,10] + 9.836381919022900199967236210084047918127677209354580364785826568010653044652657e+01) < 100*norm(L[10,10])*eps()
    @test norm(L[1000,1000] + 1.670424124232011223460118736778899327539655416751949620343795973043019808627733e+05) < 100*norm(L[1000,1000])*eps()
end
