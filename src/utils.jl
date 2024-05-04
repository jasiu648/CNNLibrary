σ(x) = one(x) / (one(x) + exp(-x))
tanh(x) = 2.0 / (one(x) + exp(-2.0x)) - one(x)
center_diff(f, x₀; Δx=1e-3) = ( f(x₀ + Δx) - f(x₀ - Δx) )/2Δx


logsoftmax(x; dims = 1) = x .- log.(sum(exp.(x), dims = dims))

function logitcrossentropy(ŷ, y; dims = 1, agg = mean)
    _check_sizes(ŷ, y)
    agg(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))
  end

function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y)) 
     size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
        "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
      ))
    end
end
  _check_sizes(ŷ, y) = nothing 


using Random

function uniform_rand(dims::Integer...; gain::Real=1)
  scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
  (rand(Random.default_rng(), Float32, dims...) .- 0.5f0) .* scale
end

nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])