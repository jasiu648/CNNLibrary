include("structures.jl")

conv(x::GraphNode, w::GraphNode) = BroadcastedOperator(conv, x, w)

function forward(::BroadcastedOperator{typeof(conv)}, x, w)
 
    # get dimensions
    (H, W, C) = size(x)
    (FH, FW, _, K) = size(w)

    # calculate output dimensions
    out_h = H - FH + 1
    out_w = W - FW + 1

    x_pad = zeros(H, W, C)
    x_pad[1:end, 1:end, :] = x

    # initialize output
    # NOTE!: this is a 4D array, but we only use the first 3 dimensions
    out = zeros(out_h, out_w, K, 1)

    # perform convolution
    for i ∈ 1:out_h
        for j ∈ 1:out_w
            # get receptive field
            r_field =
                x_pad[(i-1)+1:(i-1)+FH, (j-1)+1:(j-1)+FW, :, :]

            # flatten receptive field and weights
            r_field_flat = reshape(r_field, FH * FW * C, :)
            w_flat = reshape(w, FH * FW * C, K)

            # calculate output for this location
            out[i, j, :] = sum(w_flat .* r_field_flat, dims = 1)
        end
    end
    return out
end

function backward(::BroadcastedOperator{typeof(conv)}, x, w, g)

    # get dimensions
    (H, W, C) = size(x)
    (FH, FW, _, K) = size(w)

    # calculate output dimensions
    out_h = H - FH + 1
    out_w = W - FW + 1

    x_pad = zeros(H, W, C)
    x_pad[1:end, 1:end, :] = x

    # initialize gradients
    gx_pad = zeros(H, W, C)
    gw = zeros(size(w))

    # perform backward pass
    for i ∈ 1:out_h
        for j ∈ 1:out_w
            # get receptive field
            r_field =
                x_pad[(i-1)+1:(i-1)+FH, (j-1)+1:(j-1)+FW, :, :]

            # flatten receptive field and weights
            r_field_flat = reshape(r_field, FH * FW * C, :)
            w_flat = reshape(w, FH * FW * C, K)

            # calculate gradients for this location

            dout_local = reshape(g[i, j, :], K, 1)
            field_dout_prod = r_field_flat * dout_local'
            field_dout_prod = reshape(field_dout_prod, FH, FW, C, K)
            gw += field_dout_prod

            flat_dout_prod = w_flat * dout_local
            flat_dout_prod = reshape(flat_dout_prod, FH, FW, C, :)
            gx_pad[(i-1)+1:(i-1)+FH, (j-1)+1:(j-1)+FW, :, :] += flat_dout_prod
        end
    end

    # remove padding from gx
    gx = gx_pad[1:end, 1:end, :]

    return tuple(gx, gw)
end