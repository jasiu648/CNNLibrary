include("structures.jl")

maxpool(x::GraphNode) = BroadcastedOperator(maxpool, x)
forward(node::BroadcastedOperator{typeof(maxpool)}, x) =
    let
        input_height, input_width, layers = size(x)

        output_height = div(input_height, 2)
        output_width = div(input_width, 2)

        y = zeros(output_height, output_width, layers)
        indexes = CartesianIndex{3}[]
        for c in 1:layers
            for j in 1:output_height
                for i in 1:output_width
                    value, ids = findmax(@view x[2*j-1:2*j, 2*i-1:2*i, c])
                    y[j, i, c] = value

                    ids, idy = ids[1] + 2 * j - 1 - 1, ids[2] + 2 * i - 1 - 1
                    push!(indexes, CartesianIndex(ids, idy, c))
                end
            end
        end
        node.cache = indexes
        return y
    end

backward(node::BroadcastedOperator{typeof(maxpool)}, x, g) =
    let
        output = zeros(size(x))
        output[node.cache] = vcat(g...)
        tuple(output)
    end

