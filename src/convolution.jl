include("structures.jl")
using Tullio

conv(x::GraphNode, w::GraphNode) = BroadcastedOperator(conv, x, w)

function forward(::BroadcastedOperator{typeof(conv)}, x, w)
 
    (H, W, C) = size(x)
    (FH, FW, _, K) = size(w)

    out_h = H - FH + 1
    out_w = W - FW + 1

    out = zeros(out_h, out_w, K)

    #=
    for i ∈ 1:out_h
        for j ∈ 1:out_w
            r_field = x[(i-1)+1:(i-1)+FH, (j-1)+1:(j-1)+FW]
            r_field_flat = reshape(r_field, FH * FW * C, :)
            w_flat = reshape(w, FH * FW * C, K)
            out[i, j, :] = sum(w_flat .* r_field_flat, dims = 1)
        end
    end
    return out
    =#

    @tullio out[i, j, k] = sum(w[fx, fy, c, k] * x[i+fx-1, j+fy-1, c] for fx in 1:FH, fy in 1:FW, c in 1:C)

    return reshape(out, out_h, out_w, K, 1)
end

function backward(::BroadcastedOperator{typeof(conv)}, x, w, g)

    (H, W, C) = size(x)
    (FH, FW, _, K) = size(w)

    out_h = H - FH + 1
    out_w = W - FW + 1

    gw = zeros(size(w))

    #=
    for i ∈ 1:out_h
        for j ∈ 1:out_w

            
            r_field = x[(i-1)+1:(i-1)+FH, (j-1)+1:(j-1)+FW, :, :]
            r_field_flat = reshape(r_field, FH * FW * C, :) #add to cache

            dout_local = reshape(g[i, j, :], K, 1)
            field_dout_prod = r_field_flat * dout_local'
            field_dout_prod = reshape(field_dout_prod, FH, FW, C, K)
            gw += field_dout_prod
            #=
            flat_dout_prod = w_flat * dout_local
            flat_dout_prod = reshape(flat_dout_prod, FH, FW, C, :)
            gx[(i-1)+1:(i-1)+FH, (j-1)+1:(j-1)+FW, :, :] += flat_dout_prod
            =#
            
            @tullio gw[fx, fy, c, k] += x[i+fx-1, j+fy-1, c] * g[i, j, k, 1] (i in 1:out_h, j in 1:out_w, fx in 1:FH, fy in 1:FW, c in 1:C, k in 1:K)
        end
    
    end=#
    @tullio gw[fx, fy, c, k] += x[i+fx-1, j+fy-1, c] * g[i, j, k, 1] (i in 1:out_h, j in 1:out_w, fx in 1:FH, fy in 1:FW, c in 1:C, k in 1:K)
    return tuple(nothing, gw)
end