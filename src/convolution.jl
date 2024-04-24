include("structures.jl")

conv(x::GraphNode, w::GraphNode) = BroadcastedOperator(conv, x, w)

function forward(::BroadcastedOperator{typeof(conv)}, x, w)
    # default values 
    # NOTE: Should be same as in backward
    padding = 0
    stride = 1
    # get dimensions
    (H, W, C, _) = size(x)
    (FH, FW, _, K) = size(w)

    # calculate output dimensions
    out_h = Int(floor((H + 2 * padding - FH) / stride)) + 1
    out_w = Int(floor((W + 2 * padding - FW) / stride)) + 1

    # pad input
    p = padding
    x_pad = zeros(H + 2p, W + 2p, C)
    x_pad[p+1:end-p, p+1:end-p, :] = x

    # initialize output
    # NOTE!: this is a 4D array, but we only use the first 3 dimensions
    out = zeros(out_h, out_w, K, 1)

    # perform convolution
    for i ∈ 1:out_h
        for j ∈ 1:out_w
            # get receptive field
            r_field =
                x_pad[(i-1)*stride+1:(i-1)*stride+FH, (j-1)*stride+1:(j-1)*stride+FW, :, :]

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
    # default values 
    # NOTE: Should be same as in forward
    padding = 0
    stride = 1

    # get dimensions
    (H, W, C, _) = size(x)
    (FH, FW, _, K) = size(w)

    # calculate output dimensions
    out_h = Int(floor((H + 2 * padding - FH) / stride)) + 1
    out_w = Int(floor((W + 2 * padding - FW) / stride)) + 1

    # pad input
    p = padding
    x_pad = zeros(H + 2p, W + 2p, C)
    x_pad[p+1:end-p, p+1:end-p, :] = x

    # initialize gradients
    gx_pad = zeros(H + 2p, W + 2p, C)
    gw = zeros(size(w))

    # perform backward pass
    for i ∈ 1:out_h
        for j ∈ 1:out_w
            # get receptive field
            r_field =
                x_pad[(i-1)*stride+1:(i-1)*stride+FH, (j-1)*stride+1:(j-1)*stride+FW, :, :]

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
            gx_pad[(i-1)*stride+1:(i-1)*stride+FH, (j-1)*stride+1:(j-1)*stride+FW, :, :] +=
                flat_dout_prod
        end
    end

    # remove padding from gx
    gx = gx_pad[p+1:end-p, p+1:end-p, :]

    return tuple(gx, gw)
end