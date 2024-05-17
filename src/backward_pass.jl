include("structures.jl")

update!(node::Constant, gradient) = nothing
#=
update!(node::GraphNode, gradient) = begin
    if typeof(node) == Variable
        if isnothing(node.gradient)
            node.gradient = gradient
        else
            node.gradient .+= gradient
        end
        return
    end
    node.gradient = gradient
end
=#
update!(node::GraphNode, gradient) = begin
    node.gradient = gradient
    if typeof(node) == Variable
        if isnothing(node.__gradient)
            node.__gradient = gradient * 0
        else
            node.__gradient .+= gradient
        end
    end
end
function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end