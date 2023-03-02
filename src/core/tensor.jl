module Core

import Base.:+, Base.:-, Base.:*, Base.:^, Base.:(==)

"""
    Tensor

Main data sctrcuture for wrap tensors and ther gradients. This create a
computational graph chaining operands (Tensors) of each operation.

### Fields

- `data` -- matrix of <:Real; the values of the tensor
- `grad` -- gradient of `data`
- `requires_grad` -- flag for computing or no the gradient of the tensor
- `grad_fn` -- gradient function
- `prev` -- set with the operands used for create the tensor
- `op` -- informational string with the name of the operation
"""
mutable struct Tensor
    data::AbstractMatrix{<:Real}
    grad::Union{Nothing, Tensor}
    requires_grad::Bool
    grad_fn
    prev::Set{Tensor}
    op::String
end

"""
    Tensor(data::AbstractMatrix{<:Real}, requires_grad::Bool = false)

Basic constructor of Tensor.

### Input

- `data` -- vector, matrix or real number that correspong to `Tensor.data`
- `requires_grad` -- (optional, default `false`) flag for computing or no
    the gradient

### Note

All constructors transform their `data` in AbstractMatrix. In the case of
passing a vector, it will be transform into a column vector; for Real, a matrix
with only one element will be created.
"""
Tensor(data::AbstractMatrix{<:Real}; requires_grad::Bool = false) =
    Tensor(data, nothing, requires_grad, nothing, Set{Tensor}(), "")
Tensor(data::AbstractVector{<:Real}; requires_grad::Bool = false) =
    Tensor(hcat(data), requires_grad = requires_grad)
Tensor(data::Real; requires_grad::Bool = false) = 
    Tensor([data], requires_grad = requires_grad)

"""
    Tensor(data::AbstractMatrix{<:Real}, prev::Set{Tensor}, op::String;
        requires_grad::Bool = false)

Constructor with operands; use this when the tensor is the result of an operation.

### Input

- `data` -- matrix of reals that correspond to `Tensor.data`
- `prev` -- set of operands used in the operation
- `op` -- name of the operation
- `requires_grad` -- (optional, default: `false`) flag for computing or no
    the gradient
"""
Tensor(data::AbstractMatrix{<:Real}, prev::Set{Tensor}, op::String; 
    requires_grad::Bool = false) = Tensor(data, nothing, requires_grad, nothing, prev, op)
Tensor(data::AbstractVector{<:Real}, prev::Set{Tensor}, op::String; 
    requires_grad::Bool = false) = Tensor(hcat(data), prev, op, requires_grad = requires_grad)
Tensor(data::Real, prev::Set{Tensor}, op::String, requires_grad::Bool = false) =
    Tensor([data], prev, op, requires_grad = requires_grad)

"Equality comparator. Compare `Tensor.data` of each tensor"
function Base.:(==)(x::Tensor, y::Tensor)::Bool
    return x.data == y.data
end

"Aproximation comparator using relative tolerance or absolute tolerance"
function isapprox(x::Tensor, y::Tensor; rtol = 1e-05, atol = 1e-08)::Bool
    return isapprox(x.data, y.data, rtol = rtol, atol = atol)
end

"Negative unitary operator"
function Base.:-(x::Tensor)::Tensor
    out = Tensor(-x.data, Set([x]), "-", requires_grad = x.requires_grad)

    if x.requires_grad
        out.grad_fn = function()
            x.grad == nothing ? x.grad = -out.grad : x.grad += -out.grad
        end
    end

    return out
end

"Return the sum of two tensors. The size of both tensor must be the same"
function Base.:+(x::Tensor, y::Tensor)::Tensor
    requires_grad = x.requires_grad || y.requires_grad
    out = Tensor(x.data + y.data, Set([x, y]), "+", requires_grad = requires_grad)

    if requires_grad
        out.grad_fn = function()
            if x.requires_grad
                x.grad == nothing ? x.grad = out.grad : x.grad += out.grad
            end
            if y.requires_grad
                y.grad == nothing ? y.grad = out.grad : y.grad += out.grad
            end
        end
    end

    return out
end

"Return the difference between two tensors. The size of both tensors must be the same"
function Base.:-(x::Tensor, y::Tensor)::Tensor
    requires_grad = x.requires_grad || y.requires_grad
    out = Tensor(x.data - y.data, Set([x, y]), "-", requires_grad = requires_grad)

    if requires_grad
        out.grad_fn = function()
            if x.requires_grad
                x.grad == nothing ? x.grad = out.grad : x.grad += out.grad
            end
            if y.requires_grad
                y.grad == nothing ? y.grad = -out.grad : y.grad += -out.grad
            end
        end
    end

    return out
end

"""
Return the matrix product of two tensors. The tensors must be compatible under matrix
multiplication, i.e., if X is a (m, k) matrix, then Y must be a (k, n) matrix.
"""
function Base.:*(x::Tensor, y::Tensor)::Tensor
    requires_grad = x.requires_grad || y.requires_grad
    out = Tensor(x.data * y.data, Set([x, y]), "*", requires_grad = requires_grad)

    if requires_grad
        out.grad_fn = function()
            if x.requires_grad
                x.grad == nothing ? x.grad = out.grad * Tensor(y.data') : 
                    x.grad += out.grad * Tensor(y.data')
            end
            if y.requires_grad
                y.grad == nothing ? y.grad = Tensor(x.data') * out.grad :
                    y.grad += Tensor(x.data') * out.grad
            end
        end
    end

    return out
end

"Return the n-th power of the tensor."
function Base.:^(x::Tensor)::Tensor
    out = Tensor(x.data .^ n, Set([x]), "^", requires_grad = x.requires_grad)

    if x.requires_grad
        out.grad_fn = function()
            derivative = Tensor(n .* x.data .^(n - 1))
            x.grad == nothing ? x.grad = out.grad * derivative :
                x.grad += out.grad * derivative
        end
    end

    return out
end

"Return the Hadamard product of two tensors. The tensors must have the same size."
function ⊙(x::Tensor, y::Tensor)::Tensor
    requires_grad = x.requires_grad || y.requires_grad
    out = Tensor(x.data .* y.data, Set([x, y]), "⊙", requires_grad = requires_grad)

    if requires_grad
        out.grad_fn = function()
            if x.requires_grad
                x.grad == nothing ? x.grad = Tensor(out.grad.data .* y.data) :
                    x.grad += Tensor(out.grad.data .* y.data)
            end
            if y.requires_grad
                y.grad == nothing ? y.grad = Tensor(out.grad.data .* x.data) :
                    y.grad += Tensor(out.grad.data .* x.data)
            end
        end
    end

    return out
end

end