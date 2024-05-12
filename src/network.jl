include("forward_pass.jl")
include("backward_pass.jl");
include("utils.jl");
include("convolution.jl");
include("graph_building.jl");
include("load_data.jl");
include("scalar_operators.jl");
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
        if isa(node, Variable) && hasproperty(node, :accumulated_gradient)
              node.accumulated_gradient ./= batch_size
              node.output -= lr * node.accumulated_gradient
              node.accumulated_gradient .= 0
       end
    end
end

function train_model(model, x_train, y_train, batchsize, learning_rate)

    data_size = size(x_train, 4)
    epochs = 3
    
    for epoch in 1:epochs
         
        epoch_loss = 0.0
        
        @time for i = 1:data_size
            x = x_train[:,:,:, i]
            y = reshape(y_train[:, i],1,10)

            model[3].inputs = (Constant(x), model[3].inputs[2])
            model[13].inputs = (model[13].inputs[1],Constant(y))

            epoch_loss += forward!(model)

            backward!(model)

            if i % batchsize == 0
                update_weights!(model, learning_rate, batchsize)
            end
        end
        println("Epoch loss: ", epoch_loss / data_size)
    end
end

function test_model(model, x_test, y_test)

    global accurate
    global data_count
    accurate = 0
    data_count = 0

    for i = 1:40000
        x = x_test[:,:,:, i]
        y = reshape(y_test[:, i],1,10)

        model[3].inputs = (Constant(x), model[3].inputs[2])
        model[13].inputs = (model[13].inputs[1],Constant(y))

        forward!(model)
    end
    println("ACCURACY: ", accurate/data_count)
end

function conv(w, b, x, activation)
	out = conv(x, w) .+ b
	return activation(out)
end

function conv(w, x, activation)
	out = conv(x, w)
	return activation(out)
end

dense(w, b, x, activation) = activation((x * w) .+ b)
dense(w, x, activation) = activation((x * w))
dense(w, x) = x * w


function build_graph()
    
    input_size = 28
	input_channels = 1
	out_channels = 6
    kernel_size = 3

    x = Constant(uniform_rand(input_size, input_size, input_channels, 1))
    wh1 = Variable(xavier_glorot_init(input_channels, out_channels, kernel_size), name = "wh1")
    wh2 = Variable(randn(13*13*6, 84), name = "wh2")
    wo = Variable(randn(84, 10), name = "wo")
    y = Constant(randn(1,10))

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

function build_graph_advanced()
    
    input_size = 28
	input_channels = 1
	out_channels = 6

    x = Constant(uniform_rand(input_size, input_size, input_channels, 1))
    wh1 = Variable(init_kernel(input_channels, out_channels), name = "wh1")
    wh2 = Variable(init_kernel(6, 16), name = "wh2")
    wh3 = Variable(randn(400, 84), name = "wh3")
    wo = Variable(randn(84, 10), name = "wo")

    b1
    b2
    b3
    b4

    y = Constant(randn(1,10))

    x1 = conv(wh1, x, relu)
    x1.name = "x1"
  
    x2 = maxpool(x1)
    x2.name = "x2"

    x3 = conv(wh2, x2, relu)
    x3.name = "x3"

    x4 = maxpool(x3)
    x4.name = "x4"

    x5 = flatten(x4)
    x5.name = "x5"
    
    x6 = dense(wh2, x3, relu)
    x6.name = "x4"

    x7 = dense(wo, x6)
    x7.name = "x7"

    x8 = cross_entropy_loss(x7, y)
    x8.name = "x8"
    
    return topological_sort(x6)


end