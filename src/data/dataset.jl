import ..Jacobi
import ..Jacobi: Tensor as Tensor
import Random

struct Dataset
    X::Tensor
    y::Tensor
    batch::Int64

    Dataset(X::Tensor, y::Tensor, batch::Int64) = new(X, y, batch)
    Dataset(X::Tensor, y::Tensor) = new(X, y, size(X)[begin])
end

function Base.length(dataset::Dataset)
    data_size = size(dataset.X)[1]
    return (data_size ÷ dataset.batch) + (data_size % dataset.batch != 0)
end

function Base.iterate(dataset::Dataset, state = 1)
    if state <= length(dataset)
        start_batch = 1 + (dataset.batch * (state - 1))
        end_batch = min(start_batch + dataset.batch - 1, size(dataset.X)[1])
        return (
            (dataset.X[start_batch:end_batch, :], dataset.y[start_batch:end_batch]), 
            state + 1)
    end
    return nothing
end

function shuffle(dataset::Dataset)
    idx = Random.randperm(size(dataset.X)[1])
	return Dataset(Tensor(dataset.X[idx, :]), Tensor(dataset.y[idx]), dataset.batch)
end