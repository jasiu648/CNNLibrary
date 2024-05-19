using Random

function uniform_rand(dims::Integer...; gain::Real=1)
  scale = Float32(gain) * sqrt(24 / sum(nfan(dims...)))
  (rand(Random.default_rng(), Float32, dims...) .- 0.5f0) .* scale
end

nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])

function init_kernel(n_input::Int64, n_output::Int64; kernel_size = 3)
  stddev = sqrt(1 / (n_input * 9))
  return stddev .- rand(kernel_size, kernel_size, n_input, n_output) * stddev * 2
end

function xavier_glorot_init(in_channels, out_channels, kernel_size)
  fan_in = in_channels * kernel_size^2
  fan_out = out_channels * kernel_size^2
  limit = sqrt(6 / (fan_in + fan_out))
  return limit * randn(kernel_size, kernel_size, in_channels, out_channels)
end
