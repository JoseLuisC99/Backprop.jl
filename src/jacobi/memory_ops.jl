Base.convert(::Type{<:Numeric}, x::Tensor) = x.data

function Base.getindex(x::Tensor, keys...)::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad, 
        function (out::Tensor)
            _grad = Base.zeros(size(x))
            _grad[keys...] = out.data
            return x.autograd.requires_grad ? Tensor(_grad) : nothing
        end)
    return Tensor(x.data[keys...], (x, ), autograd, "getindex")
end

function Base.reshape(x::Tensor, shape::DimsArg)::Tensor
    @assert 0 ∉ shape "zeros not allowed in shape"
    @assert sum(-1 .== shape) <= 1 "only one dimension can be inferred"
    for d in shape
        @assert d >= -1 "invalid shape dimension $d"
    end

    new_shape = Tuple(d == -1 ? Int64(-prod(size(x)) / prod(shape)) : d for d in shape)
    autograd = AutogradMetadata(x.autograd.requires_grad,
        function (out::Tensor)
            return x.autograd.requires_grad ? reshape(out, size(x)) : nothing
        end)
    return Tensor(reshape(x.data, new_shape), (x, ), autograd, "reshape_$(x.name)")
end

# function expand(x::Tensor, sizes::Tuple{Vararg{Integer}})::Tensor
#     @assert length(sizes) >= length(size(x))
# 	original_size = Base.ones(Int64, length(sizes))
# 	original_size[(length(sizes) - length(size(x)) + 1):end] .= size(x)

#     repetitions = [new_size ÷ old_size for (new_size, old_size) in zip(sizes, original_size)]

# 	autograd = AutogradMetadata(x.autograd.requires_grad,
#         function (out::Tensor)
# 			g = sum(out.data, dims=original_size)
# 			g = reshape(g, size(x))
#             return x.autograd.requires_grad ? Tensor(g) : nothing
#         end)
	
#     tensor_data = repeat(x.data, repetitions...)
#     tensor_data = reshape(tensor_data, sizes)

#     return Tensor(tensor_data, (x, ), autograd, "expand_$(x.name)")
# end

function Base.repeat(x::Tensor, repeats::Tuple{Vararg{Integer}})::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad, 
        function (out::Tensor)
            grad = Base.zeros(length(x))
            indices = reshape(collect(1:length(x)), size(x))
            indices = repeat(indices, repeats...)

            for i in 1:length(x)
                grad[i] = sum(out[indices .== i])
            end
            return x.autograd.requires_grad ? Tensor(reshape(grad, size(x))) : nothing
        end)
    return Tensor(repeat(x.data, repeats...), (x, ), autograd, "repeat_$(x.name)")
end

function Base.permutedims(x::Tensor, perm::Tuple{Vararg{Integer}})::Tensor
    autograd = AutogradMetadata(x.autograd.requires_grad,
        function (out::Tensor)
            return x.autograd.requires_grad ? Base.permutedims(out, invperm(perm)) : nothing
        end)
    return Tensor(Base.permutedims(x.data, perm), (x, ), autograd, "perm_$(x.name)")
end

function flip(x::Tensor)::Tensor
end

function pad(x::Tensor)::Tensor
end

function shrink(x::Tensor)::Tensor
end