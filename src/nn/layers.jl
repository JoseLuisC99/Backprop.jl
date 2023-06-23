import ..Jacobi
import ..Jacobi: Tensor as Tensor

abstract type AbstractLayer end

struct Layer <: AbstractLayer
    call::Function
    params::Tuple{Vararg{<:Tensor}}
end

function Linear(in_features::Int64, out_features::Int64, affine::Bool = true)::Layer
    weight = Jacobi.randn((in_features, out_features), requires_grad = true)
    bias = affine ? Jacobi.randn((1, out_features), requires_grad = true) : nothing
    
    function call(x::Tensor)
        output = x * weight
        if affine
            output = output + bias
        end
        return output
    end

    return Layer(call, affine ? (weight, bias) : (weight, ))
end

function ReLU()::Layer
    function call(x::Tensor)
        return Jacobi.relu(x)
    end

    return Layer(call, ())
end