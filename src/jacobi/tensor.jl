using LinearAlgebra
import Random
import Base.:(==)
import Base.Broadcast: broadcasted, materialize

const Numeric{T<:Number} = Union{T, AbstractArray{<:T}}
const DimsArg = Union{Integer, Tuple{Vararg{Integer}}}

abstract type AbstractTensor end

mutable struct AutogradMetadata
    requires_grad::Bool
    grad_fn::Union{Nothing, Function}
    grad_accumulator::Union{Nothing, Function}

    AutogradMetadata() = new(false, nothing, nothing)
    AutogradMetadata(requires_grad::Bool) = 
        new(requires_grad, nothing, Base.:+)
    AutogradMetadata(requires_grad::Bool, grad_fn::Function) =
        new(requires_grad, grad_fn, Base.:+)
end

mutable struct Tensor <: AbstractTensor
    data::Numeric
    grad::Union{Nothing, AbstractTensor}
    ctx::Tuple{Vararg{<:AbstractTensor}}
    autograd::AutogradMetadata
    name::String

    Tensor(data::Numeric, ctx::Tuple{Vararg{<:AbstractTensor}} = (); requires_grad::Bool=false, name::String="") = 
        new(data, nothing, ctx, AutogradMetadata(requires_grad), name)
    Tensor(data::Numeric, ctx::Tuple{Vararg{<:AbstractTensor}}, autograd::AutogradMetadata, name::String="") = 
        new(data, nothing, ctx, autograd, name)
end

struct TensorFunction
    op::String
	forward::Function
	backward::Function
end

Base.:(==)(x::Tensor, y::Tensor)::Bool = x.data == y.data
Base.isapprox(x::Tensor, y::Tensor; kwargs...)::Bool = isapprox(x.data, y.data; kwargs...)

Base.ndims(x::Tensor) = ndims(x.data)
Base.size(x::Tensor) = size(x.data)
Base.size(x::Tensor, dim::Integer) = size(x.data, dim)
Base.length(x::Tensor) = prod(size(x))
Base.IndexStyle(::Type{<:AbstractTensor}) = IndexLinear()

Base.firstindex(x::Tensor) = 1
Base.lastindex(x::Tensor) = lastindex(x.data)
Base.firstindex(x::Tensor, dim::Integer) = 1
Base.lastindex(x::Tensor, dim::Integer) = lastindex(x.data, dim)
Base.eltype(x::Tensor) = Base.eltype(x.data)

function Base.iterate(x::Tensor)
    @inbounds return Base.length(x) == 0 ? nothing : (x[1], 2)
end

function Base.iterate(x::Tensor, state::Integer)
    @inbounds return state > Base.length(x) ? nothing : (x[state], state + 1)
end

function ones(dims::DimsArg; eltype::Type{D}=Float64, kwargs...) where D<:Number
    return Tensor(Base.ones(D, dims); name="t_ones", kwargs...)
end

function zeros(dims::DimsArg; eltype::Type{D}=Float64, kwargs...) where D<:Number
    return Tensor(Base.zeros(D, dims); name="t_zeros", kwargs...)
end

function fill(value::Number, dims::DimsArg; kwargs...)
    return Tensor(Base.fill(value, dims); name="t_const($value)", kwargs...)
end

ones_like(x::Tensor; kwargs...) = ones(size(x); kwargs...)
zeros_like(x::Tensor; kwargs...) = zeros(size(x); kwargs...)
fill_like(value::Number, x::Tensor; kwargs...) = fill(value, size(x); kwargs...)

function eye(dims::DimsArg; eltype::Type{D}=Float64, kwargs...)::Tensor where D<:Number
    if typeof(dims) <: Integer
        dims = (dims, dims)
    end
	return Tensor(Matrix{D}(I, dims); name="t_eye", kwargs...)
end

function arange(stop::Integer; eltype::Type{D}=Float64, start::Integer = 0, step::Integer = 1, kwargs...)::Tensor where D<:Number
	return Tensor(collect(D, range(start, stop, step=step)); name="t_arange($start, $step, $stop)", kwargs...)
end

function rand(dims::DimsArg; eltype::Type{D}=Float64, rng::Random.AbstractRNG=Random.MersenneTwister(), kwargs...)::Tensor where D<:Number
    return Tensor(Random.rand(rng, D, dims...); name="t_rand", kwargs...)
end

function randn(dims::DimsArg; eltype::Type{D}=Float64, rng::Random.AbstractRNG=Random.MersenneTwister(), kwargs...)::Tensor where D<:Number
    return Tensor(Random.randn(rng, D, dims...); name="t_randn",  kwargs...)
end

function randexp(dims::DimsArg; eltype::Type{D}=Float64, rng::Random.AbstractRNG=Random.MersenneTwister(), kwargs...)::Tensor where D<:Number
    return Tensor(Random.randexp(rng, D, dims...); name="t_randexp",  kwargs...)
end

function pullback_eval_seq(x::Tensor)::Vector{Tensor}
    function topological_sort(node, dag, visited)
        if node ∉ visited
            push!(visited, node)
            for v ∈ node.ctx
                topological_sort(v, dag, visited)
            end
            push!(dag, node)
        end
    end

    dag = Vector{Tensor}()
    visited = Set{Tensor}()
    topological_sort(x, dag, visited)
	return dag[end:-1:begin]
end

function backward!(x::Tensor)::Nothing
    graph = pullback_eval_seq(x)
    x.grad = ones(size(x), name="grad")

    for tensor in graph
        !any(t.autograd.requires_grad for t ∈ tensor.ctx) && continue
        
        @assert (tensor.grad !== nothing) "Null gradient on tensor variable"
        out_grads = tensor.autograd.grad_fn(tensor.grad)
        if typeof(out_grads) <: Tensor
            out_grads = (out_grads, )
        end
        for (t, g) in zip(tensor.ctx, out_grads)
            if g !== nothing && t.autograd.requires_grad
                # @assert (size(t) == size(g)) "gradient dimension $(size(g)) must match tensor dimension $(size(t))"
                t.grad === nothing ? t.grad = g : t.grad = t.autograd.grad_accumulator(t.grad, g)
            end
        end
    end
    return
end

function zero_grad(x::Tensor)::Nothing
    graph = pullback_eval_seq(x)
    for tensor in graph
        if tensor.autograd.requires_grad
            tensor.grad = zeros_like(tensor)
        end
    end
    return
end

function clear_grads(x::Tensor)::Nothing
    graph = pullback_eval_seq(x)
    for tensor in graph
        if tensor.autograd.requires_grad
            tensor.grad = nothing
        end
    end
    return
end

function apply(tfunc::TensorFunction, args...)::Tensor
    tensor_args = filter(x -> x isa Tensor, args)
    # if any(t.autograd.requires_grad for t ∈ tensor_args)
    #     autograd = AutogradMetadata(true, tfunc.backward(args...))
    #     return Tensor(tfunc.forward(args...), tensor_args, autograd, tfunc.op)
    # else
    #     return Tensor(tfunc.forward(args...), tensor_args, name=tfunc.op)
    # end
    autograd = AutogradMetadata(true, tfunc.backward(args...))
    return Tensor(tfunc.forward(args...), tensor_args, autograd, tfunc.op)
end

# function Base.show(io::IO, tensor::Tensor)
#     print(io, "Tensor $(tensor.name) (\ndata = ")
#     Base.show(io, "text/plain", tensor.data)
#     print(io, "\ngrad = ")
#     Base.show(io, "text/plain", tensor.grad === nothing ? nothing : tensor.grad.data)
#     println(io, ")")
# end

# function broadcasted(f::Function, A::Tensor, B::Tensor)
#     autograd = AutogradMetadata(A.autograd.requires_grad || B.autograd.requires_grad, 
#         function (out::Tensor)
#             return A.autograd.requires_grad ? out : nothing,
#                 B.autograd.requires_grad ? out : nothing
#         end)
#     println("A size: $(size(A))")
#     println("B size: $(size(B))")
#     return Tensor(f.(A.data, B.data), (A, B), autograd, "broadcasted \\n($(Base.nameof(f)))")
# end

function broadcasted(f::Function, A::Tensor, B::Tensor)
    if size(A) == size(B)
        return f(A, B)
    end
	
    broadcast_ndims = max(ndims(A), ndims(B))
    if ndims(A) != broadcast_ndims
        new_dims = Tuple([Base.ones(Int64, broadcast_ndims - ndims(A)); size(A)...])
        A = reshape(A, new_dims)
    end
    if ndims(B) != broadcast_ndims
        new_dims = Tuple([Base.ones(Int64, broadcast_ndims - ndims(B)); size(B)...])
        B = reshape(B, new_dims)
    end

	repeats = Tuple(max(da, db) ÷ min(da, db) for (da, db) in zip(size(A), size(B)))
	final_dims = Tuple(max(da, db) for (da, db) in zip(size(A), size(B)))

	if size(A) != final_dims
		A = repeat(A, repeats)
	end
	if size(B) != final_dims
		B = repeat(B, repeats)
	end

	return f(A, B)
end

function materialize(::Type{Tensor}, x)
    println("materialize!")
    return Tensor(x)
end