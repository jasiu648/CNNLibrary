include("./dualnumbers.jl")

ReLU(x) = max(zero(x), x)
σ(x) = one(x) / (one(x) + exp(-x))
tanh(x) = 2.0 / (one(x) + exp(-2.0x)) - one(x)
ϵ = Dual(0., 1.)
center_diff(f, x₀; Δx=1e-3) = ( f(x₀ + Δx) - f(x₀ - Δx) )/2Δx
D = derivative(f, x) = partials(f(Dual(x, one(x))))

J = function jacobian(f, args::Vector{T}) where {T <:Number}
    jacobian_columns = Matrix{T}[]
    
    for i=1:length(args)
        x = Dual{T}[]
        for j=1:length(args)
            push!(x, (i == j) ? Dual(args[j], one(args[j])) :
                                Dual(args[j],zero(args[j])) )
        end
        column = partials.([f(x)...])
        push!(jacobian_columns, column[:,:])
    end
    hcat(jacobian_columns...)
end

H = function hessian(f, args::Vector)
    ∇f(x::Vector) = J(f, x)
    J(∇f, args)
end

using Statistics: mean  # standard library
function loss_and_accuracy(model, data)
    (x,y) = only(loader(data; batchsize=length(data)))
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    (; loss, acc, split=data.split)  # return a NamedTuple
end

function logit_cross_entropy(ŷ, y)
    # Ensure inputs are column vectors
    ŷ = reshape(ŷ, :, 1)
    y = reshape(y, :, 1)
    
    # Compute cross-entropy loss
    n = size(ŷ, 1)
    loss = -(y' * log.(ŷ) + (1 .- y)' * log.(1 .- ŷ)) / n
    
    return loss[1]
end