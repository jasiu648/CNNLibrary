include("structures.jl")


# Building graph
function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end


# Updating gradient
update!(node::Constant, gradient) = nothing

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


# Backward pass
function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
end

backward!(node::Constant) = nothing
backward!(node::Variable) = nothing

function backward!(node::Operator)
    inputs = node.inputs
    input_gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, input_gradient) in zip(inputs, input_gradients)
        update!(input, input_gradient)
    end
    return nothing
end


# Forward pass
compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector{<:GraphNode})
    for node in order
        compute!(node)
    end
    return last(order).output
end
