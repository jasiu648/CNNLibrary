include("structure.jl")
import Base: ^, sin, sum, *, +, -, max, reshape
import LinearAlgebra: mul!, diagm
using Tullio

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(x, zero(x))
backward(::BroadcastedOperator{typeof(relu)}, x, g) = return tuple(g .* (x .> 0))

flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, :)
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = tuple(reshape(g, size(x)))

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
        return mean(-sum(y .* (y_hat .- log.(sum(exp.(y_hat))))))
    end
backward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat - y))
    end

convolution(x::GraphNode, kernel::GraphNode) = BroadcastedOperator(convolution, x, kernel)
forward(::BroadcastedOperator{typeof(convolution)}, x, w) =
    let
        (H, W, C) = size(x)

    (FH, FW, _, K) = size(w)

    out_h = H - FH + 1
    out_w = W - FW + 1

    out = zeros(out_h, out_w, K)

    @tullio out[i, j, k] = sum(w[fx, fy, c, k] * x[i+fx-1, j+fy-1, c] for fx in 1:FH, fy in 1:FW, c in 1:C)

    return reshape(out, out_h, out_w, K, 1)
    end

backward(::BroadcastedOperator{typeof(convolution)}, x, w, g) =
    let
        (H, W, C) = size(x)
        (FH, FW, _, K) = size(w)
    
        out_h = H - FH + 1
        out_w = W - FW + 1
    
        gw = zeros(size(w))

        @tullio gw[fx, fy, c, k] += x[i+fx-1, j+fy-1, c] * g[i, j, k, 1] (i in 1:out_h, j in 1:out_w, fx in 1:FH, fy in 1:FW, c in 1:C, k in 1:K)
        return tuple(nothing, gw)
    end

dense(x::GraphNode, w::GraphNode) = BroadcastedOperator(dense, x, w)
forward(::BroadcastedOperator{typeof(dense)}, x, w) = let
    return w*x
end
backward(::BroadcastedOperator{typeof(dense)}, x, w, g) = let
    tuple(w' * g, g * x')
end
    

maxpool(x::GraphNode) = BroadcastedOperator(maxpool, x)
forward(node::BroadcastedOperator{typeof(maxpool)}, x) =
    let
        h, w, c = size(x)
        output = zeros(h ÷ 2, w ÷ 2, c)
        indices = CartesianIndex{3}[]
        for i = 1:c
            for j = 1:h÷2
                for k = 1:w÷2
                    val, ids = findmax(@view x[2*j-1:2*j, 2*k-1:2*k, i])
                    output[j, k, i] = val

                    idx, idy = ids[1] + 2 * j - 1 - 1, ids[2] + 2 * k - 1 - 1
                    push!(indices, CartesianIndex(idx, idy, i))
                end
            end
        end
        node.cache = indices
        return output
    end

backward(node::BroadcastedOperator{typeof(maxpool)}, x, g) =
    let
        output = zeros(size(x))
        output[node.cache] = vcat(g...)
        tuple(output)
    end