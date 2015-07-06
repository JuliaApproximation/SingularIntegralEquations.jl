p = pwd()
cd(Pkg.dir("SingularIntegralEquations/deps/"))
run(`make`)
cd(p)
