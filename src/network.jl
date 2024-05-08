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

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && !isnothing(node.gradient)
              node.gradient ./= batch_size
              node.output -= lr * node.gradient
              node.gradient .= 0
       end
    end
end

function train_model(model, x_train,y_train,batchsize,learning_rate)

    for i = 1:100
    #for i = 1:size(x_train, 4)
        x = x_train[:,:,:, i]
        y = reshape(y_train[:, i],1,10)

        model[1] = Constant(x)
        model[12] = Constant(y)

        forward!(model)
        backward!(model)
        if i % batchsize == 0
            update_weights!(model, learning_rate, batchsize)
        end
    end

end

function test_model(x, y)

    for i=1:size(x_data, 4)
        x = Constant(x_data[:, :, i])
        y = Constant(y_data[i, :])
        graph = build_graph(x, y, cnn)
		forward!(graph)
    end
end

function conv(w, b, x, activation)
	out = conv(x, w) .+ b
	return activation(out)
end

function conv(w, x, activation)
	out = conv(x, w)
	return activation(out)
end

function mean_squared_loss(y, ŷ)
    return Constant(0.1) .* sum((y .- ŷ))
end

dense(w, b, x, activation) = activation((x * w) .+ b)
dense(w, x, activation) = activation((x * w))
dense(w, x) = x * w


function build_graph()
    
    input_size = 28
	kernel_size = 3
	input_channels = 1
	out_channels = 6

    x = Constant(uniform_rand(input_size, input_size, input_channels, 1))
    wh1 = Variable(uniform_rand(kernel_size, kernel_size, input_channels, out_channels), name = "wh1")
    wh2 = Variable(randn(13*13*6, 84), name = "wh2")
    wo = Variable(randn(84, 10), name = "wo")
    y = Constant(randn(1,10))
    
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
    
    x6 = cross_entropy_loss(x5, y)
    x6.name = "x6"
    
    return topological_sort(x6)
end