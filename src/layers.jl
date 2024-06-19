include("utils.jl")
include("operators.jl")
include("graph_operations.jl")

abstract type Layer end

mutable struct ConvLayer <: Layer
    weights :: Variable
    bias :: Any
    function ConvLayer(dims::Integer...; bias::Bool = false, weights_init::Function = uniform_rand)
        weights = Variable(weights_init(dims...))
        new(weights,nothing)
    end
end

mutable struct DenseLayer <: Layer
    weights :: Variable
    bias :: Any
    function DenseLayer(dims::Integer...; bias::Bool = false, weights_init::Function = uniform_rand)
        weights = Variable(weights_init(dims...))
            if bias
                new(weights, Variable(uniform_rand(dims[1],1)))
            else
                new(weights,nothing)
            end
    end
end

mutable struct MaxPoolLayer <: Layer
    window :: Constant
    function MaxPoolLayer(window :: Tuple{Int, Int})
        new(Constant(window))
    end
end

mutable struct FlattenLayer <: Layer
end

mutable struct ReLULayer <: Layer
end

mutable struct LogitCrossEntropyLayer <: Layer
end

function CreateOperator(layer::ConvLayer, input)
    return convolution(input, layer.weights)
end

function CreateOperator(layer::DenseLayer, input)
    if isnothing(layer.bias)
        return dense(input, layer.weights)
    else
        return dense(input, layer.weights, layer.bias)
    end
    
end

function CreateOperator(layer::MaxPoolLayer, input)
    return maxpool(input, layer.window)  # Add dims 
end

function CreateOperator(layer::ReLULayer, input)
    return relu(input)
end

function CreateOperator(layer::FlattenLayer, input)
    return flatten(input)
end

function CreateOperator(layer::LogitCrossEntropyLayer, input...)
    return cross_entropy_loss(input[1],input[2])
end


function build_graph(layers::Vector{Layer}, input_size, label_size)
    x = Constant(rand(input_size...))
    y = Constant(rand(label_size...))

    layers_count = size(layers,1)
    operators = Vector{GraphNode}(undef,layers_count)
    operators[1] = CreateOperator(layers[1], x)

    for i = 2:layers_count-1
        operators[i] = CreateOperator(layers[i], operators[i-1])
    end

    operators[layers_count] = CreateOperator(layers[layers_count], operators[layers_count-1], y)
    
    return ModelCNN(topological_sort(operators[layers_count]), x, y, operators[layers_count-1])
end