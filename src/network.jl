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
        if isa(node, Variable) && hasproperty(node, :__gradient)
            node.__gradient ./= batch_size
            node.output -= lr * node.__gradient
            node.__gradient .= 0
        end
    end
end
#=
function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable)
              node.output -= lr * node.gradient ./ batch_size
              node.gradient .= 0
       end
    end
end
=#
function train_model(model, x_train, y_train, batchsize, learning_rate, epochs)

    data_size = size(x_train, 4)
    
    for epoch in 1:epochs
         
        epoch_loss = 0.0
        
        @time for i = 1:data_size
            x = x_train[:,:,:, i]
            y = y_train[:, i]

            @views model[3].inputs = (Constant(x), model[3].inputs[2])
            @views model[13].inputs = (model[13].inputs[1],Constant(y))
            
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
    data_size = size(x_test, 4)

    for i = 1:data_size
        x = x_test[:,:,:, i]
        y = y_test[:, i]

        @views model[3].inputs = (Constant(x), model[3].inputs[2])
        @views model[13].inputs = (model[13].inputs[1],Constant(y))

        forward!(model)
    end
    println("ACCURACY: ", accurate/data_count)
end

function conv(w, b, x, activation)
	out = conv(x, w) .+ b
	return activation(out)
end



function build_graph()
    
    input_size = 28
	input_channels = 1
	out_channels = 6
    kernel_size = 3

    x = Constant(zeros(input_size, input_size, input_channels))
    wh1 = Variable(init_kernel(input_channels, out_channels), name = "wh1")
    wh2 = Variable(randn(84, 1014), name = "wh2")
    wo = Variable(randn(10, 84), name = "wo")
    y = Constant(zeros(10,1))

    x1 = conv(x, wh1) |> relu |> maxpool |> flatten
    x2 = dense(x1, wh2) |> relu
    x3 = dense(x2,wo)
    x4 = cross_entropy_loss(x3, y)
    
    return topological_sort(x4)
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

    b1 = Variable(zeros(6))
    b2 = Variable(zeros(16))
    b3 = Variable(zeros(84,1))
    b4 = Variable(zeros(1,10))

    y = Constant(randn(10,10))

    x1 = conv(wh1, x, relu)
    x1.name = "x1"
  
    x2 = maxpool(x1)
    x2.name = "x2"

    x3 = conv(wh2, x2)
    x3.name = "x3"

    x4 = maxpool(x3)
    x4.name = "x4"

    x5 = flatten(x4)
    x5.name = "x5"
    
    x6 = dense(wh3, x5)
    x6.name = "x4"

    x7 = dense(wo, x6)
    x7.name = "x7"

    x8 = cross_entropy_loss(x7, y)
    x8.name = "x8"
    
    return topological_sort(x8)


end