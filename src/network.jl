include("forward_pass.jl")
include("backward_pass.jl");
include("utils.jl");
include("convolution.jl");
include("graph_building.jl");
include("load_data.jl");
#include("scalar_operators.jl");
include("broadcasted_operators.jl")
include("maxpool.jl")
#=
net = Chain(
    Conv((3, 3), 1 => 6,  relu, bias=false),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(13*13*6 => 84, relu, bias=false), 
    Dense(84 => 10, identity, bias=false)
)
=#
function conv(w, b, x, activation)
	out = conv(x, w) .+ b
	return activation(out)
end

function conv(w, x, activation)
	out = conv(x, w)
	return activation(out)
end

function mean_squared_loss(y, ŷ)
    return Constant(0.1) .* sum((y .- ŷ) .^ Constant(2))
end

dense(w, b, x, activation) = activation((x * w) .+ b)
dense(w, x, activation) = activation((x * w))
dense(w, x) = x * w
function build_graph()
    
    input_size = 28
	kernel_size = 3
	input_channels = 1
	out_channels = 6

    #=
    x = Variable{Array{Float64, 4}}(uniform_rand(input_size, input_size, input_channels, 1)::Array{Float64, 4}, name = "x")
    wh1 = Variable{Array{Float64, 4}}(uniform_rand(input_size, kernel_size, input_channels, out_channels)::Array{Float64, 4}, name = "wh1")
    wh2 = Variable{Matrix{Float64}}(randn(13*13*6, 84)::Matrix{Float64}, name = "wh2")
    wo = Variable{Matrix{Float64}}(randn(84, 10)::Matrix{Float64}, name = "wo")
    y = Variable{Vector{Float64}}(randn(10,1)::Vector{Float64}, name = "y")
    =#
    x = Variable(uniform_rand(input_size, input_size, input_channels, 1), name = "x")
    wh1 = Variable(uniform_rand(kernel_size, kernel_size, input_channels, out_channels), name = "wh1")
    wh2 = Variable(randn(13*13*6, 84), name = "wh2")
    wo = Variable(randn(84, 10), name = "wo")
    wo2 = Variable(randn(10,1), name = "wo2")
    y = Variable(randn(10,1), name = "y")
    
    print(size(x.output))
    print(size(wh1.output))
    print(size(wh2.output))
    print(size(wo.output))
    print(size(y.output))

    x1 = conv(wh1, x, relu)
    x1.name = "x1"
  
    x2 = maxpool(x1)
    x2.name = "x2"

    x3 = flatten(x2)
    x3.name = "x3"

    x4 = dense(wh2, x3, relu)
    x4.name = "x4"

    x5 = dense(wo, x4)
    x5.name = "x5"
   
    x6 = dense(wo2, x5)
    x6.name = "x6"
    
    return topological_sort(x6), x, y
end