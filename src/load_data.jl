using MLDatasets

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

function onehotmatrix(labels::Vector{T}, categories::AbstractVector{T}) where T
    num_categories = length(categories)
    num_samples = length(labels)
    one_hot_matrix = zeros(T, num_categories, num_samples)
    for (i, label) in enumerate(labels)
        if label in categories
            index = findfirst(categories .== label)
            one_hot_matrix[index, i] = 1
        else
            error("Label $label not found in categories")
        end
    end
    return one_hot_matrix
  end