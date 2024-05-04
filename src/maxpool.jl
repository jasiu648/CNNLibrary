include("structures.jl")

maxpool(x::GraphNode) = BroadcastedOperator(maxpool, x)

function forward(::BroadcastedOperator{typeof(maxpool)}, x)

    input_height, input_width, layers = size(x)

    output_height = div(input_height, 2)
    output_width = div(input_width, 2)

    y = zeros(output_height, output_width, layers)
    for c in 1:layers
        for j in 1:output_height
            for i in 1:output_width
                region = x[2*j - 1:2*j, 2*i - 1:i *2, c]
                y[j, i, c] = maximum(region)
            end
        end
    end
    
    return y
end

function backward(::BroadcastedOperator{typeof(maxpool)}, x, g)
    
    output_height, output_width, layers = size(x)
    y = zeros(output_height, output_width, layers)
    
    output_height = div(output_height, 2)
    output_width = div(output_width, 2)

    for c in 1:layers
        for j in 1:output_height
            for i in 1:output_width
                region = x[2*j-1:2*j, 2*i-1:2*i,c]
                max_value = maximum(region)
                idx = findfirst(isequal(max_value), region)
                y[2*j-2 + idx[1],2*i-2 + idx[2], c] = 1
            end
        end
    end
    
    return y
end