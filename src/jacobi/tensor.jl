using LinearAlgebra
import Random: rand, randn, randexp, AbstractRNG, MersenneTwister
import Base: ones, zeros
import Base: +, -, *, ^, (==)

const Numeric{T<:Number} = Union{T, AbstractArray{<:T}}
const DimsArg{N} = Union{Integer, Tuple{Vararg{Integer, N}}}

abstract type AbstractTensor end

mutable struct AutogradMetadata
    requires_grad::Bool
    ctx::Vector{<:AbstractTensor}
    grad_fn::Union{Nothing, Function}
    grad_accumulator::Union{Nothing, Function}
    graph::Union{Nothing, Vector{<:AbstractTensor}}

    AutogradMetadata() = new(false, Vector{AbstractTensor}([]), nothing, nothing, nothing)
    AutogradMetadata(requires_grad::Bool) = 
        new(requires_grad, Vector{AbstractTensor}([]), nothing, Base.:+, nothing)
    AutogradMetadata(requires_grad::Bool, ctx::Vector{<:AbstractTensor}, grad_fn::Function) =
        new(requires_grad, ctx, grad_fn, Base.:+, nothing)
end

mutable struct Tensor <: AbstractTensor
    data::Numeric
    grad::Union{Nothing, AbstractTensor}
    autograd::AutogradMetadata

    Tensor(data::Numeric; requires_grad::Bool = false) = 
        new(data, nothing, AutogradMetadata(requires_grad))
    Tensor(data::Numeric, ctx::Vector{Tensor}; requires_grad::Bool = false) =
        new(data, nothing, AutogradMetadata(requires_grad, ctx, nothing))
    Tensor(data::Numeric, autograd::AutogradMetadata) = 
        new(data, nothing, autograd)
end

Base.ndims(x::Tensor) = ndims(x.data)
Base.size(x::Tensor) = size(x.data)
Base.size(x::Tensor, i::Int) = size(x.data, i)

Base.IndexStyle(::Type{<:Tensor}) = IndexLinear()

function ones_tensor(dims::DimsArg; eltype::Type{D}=Float64, kwargs...) where D<:Number
    return Tensor(Base.ones(D, dims); kwargs...)
end

function zeros_tensor(dims::DimsArg; eltype::Type{D}=Float64, kwargs...) where D<:Number
    return Tensor(Base.zeros(D, dims); kwargs...)
end

function eye(dims::DimsArg; eltype::Type{D}=Float64, kwargs...)::Tensor where D<:Number
    if typeof(dims) <: Integer
        dims = (dims, dims)
    end
	return Tensor(Matrix{D}(I, dims); kwargs...)
end

function arange(stop::Integer; eltype::Type{D}=Float64, start::Integer = 0, step::Integer = 1, kwargs...)::Tensor where D<:Number
	return Tensor(collect(D, range(start, stop, step=step)); kwargs...)
end

function rand_tensor(dims::DimsArg; eltype::Type{D}=Float64, rng::AbstractRNG=MersenneTwister(), kwargs...)::Tensor where D<:Number
    return Tensor(rand(rng, D, dims...); kwargs...)
end

function randn_tensor(dims::DimsArg; eltype::Type{D}=Float64, rng::AbstractRNG=MersenneTwister(), kwargs...)::Tensor where D<:Number
    return Tensor(randn(rng, D, dims...); kwargs...)
end

function randexp_tensor(dims::DimsArg; eltype::Type{D}=Float64, rng::AbstractRNG=MersenneTwister(), kwargs...)::Tensor where D<:Number
    return Tensor(randexp(rng, D, dims...))
end

function pullback_eval_seq(x::Tensor)::Vector{Tensor}
    dag = Vector{Tensor}()
    visited = Set{Tensor}([])

    function topological_sort(node, visited, dag)
        push!(visited, node)
        for v ∈ node.autograd.ctx
            if v ∉ visited
                topological_sort(v, visited, dag)
            else
                throw(CyclicGraphException())
            end
        end
        push!(dag, node)
    end

    topological_sort(x, visited, dag)
	return dag[end:-1:begin]
end

function backward(x::Tensor; retain_graph::Bool=false, allow_high_order::Bool=false)
    graph = x.autograd.graph === nothing ? pullback_eval_seq(x) : x.autograd.graph
    x.grad = ones_tensor(size(x), requires_grad=allow_high_order)

    for tensor in graph
        !any(t.autograd.requires_grad for t ∈ tensor.autograd.ctx) && continue
        
        @assert (tensor.grad !== nothing) NullGradException
        out_grads = tensor.autograd.grad_fn(tensor.grad)
        typeof(out_grads) <: Tensor ? out_grads = (out_grads, ) : nothing
        for (t, g) in zip(tensor.autograd.ctx, out_grads)
            if g !== nothing && t.autograd.requires_grad
                @assert (size(t) == size(g)) MismatchDimsException
                t.grad === nothing ? t.grad = g : t.grad = t.autograd.grad_accumulator(t.grad, g)
            end
        end
    end

    if retain_graph
        x.autograd.graph = graph
    end
end

function zero_grad(x::Tensor)::Nothing
    graph = x.autograd.graph === nothing ? pullback_eval_seq(x) : x.autograd.graph
    for tensor in graph
        if tensor.autograd.requires_grad
            tensor.grad = zeros_tensor(size(tensor))
        end
    end
end

Base.:(==)(x::Tensor, y::Tensor)::Bool = x.data == y.data
Base.isapprox(x::Tensor, y::Tensor; kwargs...)::Bool = isapprox(x.data, y.data; kwargs...)

function Base.transpose(x::Tensor)::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad, [x],
        function(out::Tensor)
            return x.autograd.requires_grad ? transpose(out.data) : nothing
        end)
    return Tensor(transpose(x.data), autograd)
end

function Base.:+(x::Tensor, y::Tensor)::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad || y.autograd.requires_grad, [x, y],
        function(out::Tensor)
            return x.autograd.requires_grad ? out : nothing, 
                y.autograd.requires_grad ? out : nothing
        end)
    return Tensor(x.data + y.data, autograd)
end

function Base.:-(x::Tensor)::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad, [x], 
        (out::Tensor) ->  x.autograd.requires_grad ? -out : nothing)
    return Tensor(-x.data, autograd)
end

function Base.:-(x::Tensor, y::Tensor)::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad || y.autograd.requires_grad, [x, y],
        function(out::Tensor)
            return x.autograd.requires_grad ? out : nothing,
                y.autograd.requires_grad ? -out : nothing
        end)
    return Tensor(x.data - y.data, autograd)
end

function Base.:*(x::Tensor, y::Tensor)::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad || y.autograd.requires_grad, [x, y], 
        function(out::Tensor)
            return x.autograd.requires_grad ? out * transpose(y) : nothing,
                y.autograd.requires_grad ? transpose(x) * out : nothing
        end)
    return Tensor(x.data * y.data, autograd)
end

function Base.:^(x::Tensor, n::Number)::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad, [x], 
        function(out::Tensor)
            local_grad = Tensor(n .* x.data .^ (n - 1))
            return x.autograd.requires_grad ? out ⊙ local_grad : nothing
        end)
    return Tensor(x.data .^ n, autograd)
end

function ⊙(x::Tensor, y::Tensor)::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad || y.autograd.requires_grad, [x, y],
        function(out::Tensor)
            return x.autograd.requires_grad ? Tensor(out.data .* y.data) : nothing, 
                y.autograd.requires_grad ? Tensor(x.data .* out.data) : nothing
        end)
    return Tensor(x.data .* y.data, autograd)
end