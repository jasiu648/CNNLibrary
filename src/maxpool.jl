include("structures.jl")

maxpool(x::GraphNode) = BroadcastedOperator(maxpool, x)

function forward(::BroadcastedOperator{typeof(maxpool)}, x)

    input_height, input_width = size(x)

    output_height = div(input_height, 2)
    output_width = div(input_width, 2)

    y = zeros(output_height, output_width)

    for j in 1:output_height
        for i in 1:output_width
            region = x[2*j - 1:2*j, 2*i - 1:i *2]
            y[j, i] = maximum(region)
        end
    end

    return y
end

function backward(::BroadcastedOperator{typeof(maxpool)}, x, p, g)
    
end