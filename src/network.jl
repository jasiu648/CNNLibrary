using Statistics
include("structures.jl")
include("operators.jl")
include("graph_operations.jl")
include("utils.jl")


function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable)
            node.gradient ./= batch_size
            node.output -= lr * node.gradient
            node.gradient .= 0
        end
    end
end
#=
function initialize_model()
    wk = Variable(uniform_rand(3,3,1, 6))
    wh =  Variable(uniform_rand(84,1014), name = "wh")
    wo = Variable(uniform_rand(10, 84), name = "wo")
    graph, input_data, label, output = build_graph(wk,wh,wo)
    return ModelCNN(graph, input_data, label, output)
end

function initialize_model2()
    wk1 = Variable(uniform_rand(3,3,1, 6))
    wk2 = Variable(uniform_rand(3,3,6,16))
    wh =  Variable(uniform_rand(84,400))
    wo = Variable(uniform_rand(10, 84))
    
    graph, input_data, label, output = build_graph2(wk1,wk2,wh,wo)
    return ModelCNN(graph, input_data, label, output)
end

function build_graph(weights_kernel::Variable, weights_hidden::Variable, weights_output::Variable)

    x = Constant(rand(28,28,1))
    y = Constant(rand(10))

    x1 = convolution(x, weights_kernel) |> relu |> maxpool |> flatten
    x2 = dense(x1, weights_hidden) |> relu
	x3 = dense(x2, weights_output)
	e = cross_entropy_loss(x3, y)
	return topological_sort(e), x, y, x3
end

function build_graph2(weights_kernel::Variable,weights_kernel2::Variable, weights_hidden::Variable, weights_output::Variable)
    x = Constant(rand(28,28,1))
    y = Constant(rand(10))
    bd2 = Variable(zeros(10,1))
    x1 = convolution(x, weights_kernel) |> relu |> maxpool
    x2 = convolution(x1, weights_kernel2) |> relu |> maxpool |> flatten
    x3 = dense(x2, weights_hidden) |> relu
	x4 = dense(x3, weights_output,bd2)
	e = cross_entropy_loss(x4, y)
    return topological_sort(e), x, y, x4
end
=#
function train_model(model::ModelCNN, trainx::Any, trainy::Any, settings::Any)
    
    train_size = size(trainx, 4)

    for epoch in 1:settings.epochs
        
        epoch_loss = 0.0
        @time for i=1:train_size
            @inbounds model.input_data.output = trainx[:, :, :, i]
            @inbounds model.label.output = trainy[i, :]
            
            epoch_loss += forward!(model.graph)
            backward!(model.graph)
            if i % settings.batchsize == 0
                update_weights!(model.graph, settings.eta, settings.batchsize)
            end
        end

        accurate = 0

        # Testing after one dataset iteration, not counted in speed time
        for i=1:train_size
            @inbounds model.input_data.output = trainx[:, :, :, i]
            @inbounds model.label.output = trainy[i, :]
            forward!(model.graph)
            accurate += argmax(model.label.output) == argmax(model.output.output)
        end

        println("Epoch:", epoch)
        println("Train accuracy: ", accurate / train_size)
        println("Epoch loss: ", epoch_loss / train_size, "\n")
    end
    return
end


function test_model(model::ModelCNN, x_data::Any, y_data::Any)

    accurate = 0
    test_size = size(x_data, 4)

    @time for i=1:test_size
        model.input_data.output = x_data[:, :, :, i]
        model.label.output = y_data[i, :]
		forward!(model.graph)
        accurate += argmax(model.label.output) == argmax(model.output.output)
    end
    println("Test Accuracy: ", accurate / test_size)
    return
end
