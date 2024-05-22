using Random

# Xavier Gloriot Wegihts initilization
function uniform_rand(dims::Integer...; gain::Real=1)
  scale = Float32(gain) * sqrt(24 / sum(nfan(dims...)))
  (rand(Random.default_rng(), Float32, dims...) .- 0.5f0) .* scale
end

nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])