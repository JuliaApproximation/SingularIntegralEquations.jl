p = pwd()
cd(Pkg.dir("SIE/deps/"))
run(`make`)
cd(p)
