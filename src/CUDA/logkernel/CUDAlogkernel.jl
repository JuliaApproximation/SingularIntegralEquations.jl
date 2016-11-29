using ApproxFun, SingularIntegralEquations, CUDA

# select a CUDA device
dev = CuDevice(0)

# create a context (like a process in CPU) on the selected device
ctx = create_context(dev)

# create the PTX module
run(`make CUDAlogkernel`)

# load the PTX module (each module can contain multiple kernel functions)
md = CuModule("CUDAlogkernel.ptx")

# retrieve the kernel function from the module
CUDAlogkernel = CuFunction(md, "CUDAlogkernel")

# generate data and load them to GPU
T = Float64
a,b = -1.,1.
u = [1.0,1/2,1/3,1/4,1/5]
n = 2^4
x = linspace(-3,3,n)
y = 0.0x+0.5
ret = Array(T,n)

@time begin
# load data arrays onto GPU
    @time gu = CuArray(u)
    @time gx = CuArray(x)
    @time gy = CuArray(y)

# create an array on GPU to store results
    @time gret = CuArray(T,n)

# run the kernel
# syntax: launch(kernel, grid_size, block_size, arguments)
# here, grid_size and block_size can be an integer or a tuple of integers
    @time launch(CUDAlogkernel, 1, n, (a, b, length(u), gu, gx, gy, gret))
    println("Second launch time: ")
    @time launch(CUDAlogkernel, 1, n, (a, b, length(u), gu, gx, gy, gret))

# download the results from GPU
    @time ret = to_host(gret)   # gret is a Julia array on CPU (host)
    println("Second retrieval time: ")
    @time to_host(gret)   # gret is a Julia array on CPU (host)

# release GPU memory
    free(gu)
    free(gx)
    free(gy)
    free(gret)

# finalize: unload module and destroy context
    unload(md)
    destroy(ctx)
end
# print the results

println("The logkernel results are: \n",ret)

sp = JacobiWeight(-.5,-.5,ChebyshevDirichlet{1,1}(Segment(a,b)))
f = Fun(sp,u)

println("The error is: ",norm(@time logkernel(f,complex(x,y))-ret))
