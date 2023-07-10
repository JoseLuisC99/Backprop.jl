import ..Jacobi
import ..Jacobi: Tensor as Tensor

abstract type AbstractLayer end

struct Layer <: AbstractLayer
    call::Function
    params::Tuple{Vararg{<:Tensor}}
end

function Linear(in_features::Int64, out_features::Int64, affine::Bool = true)::Layer
    weight = Jacobi.randn((in_features, out_features), requires_grad = true, name = "weight")
    bias = affine ? Jacobi.randn((1, out_features), requires_grad = true, name = "bias") : nothing
    
    function call(x::Tensor)
        output = x * weight
        if affine
            @assert length(x) == length(x) "data shape $(size(x)) must be compatible with bias shape $(size(bias))"
            repeats = Base.ones(Int64, ndims(x))
            repeats[begin] = size(x)[begin]
            output = output + repeat(bias, Tuple(repeats))
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

function Tanh()::Layer
    function call(x::Tensor)
        return Jacobi.tanh(x)
    end

    return Layer(call, ())
end

function Sigmoid()::Layer
    function call(x::Tensor)
        return Jacobi.sigmoid(x)
    end

    return Layer(call, ())
end

function SiLU()::Layer
    function call(x::Tensor)
        return Jacobi.silu(x)
    end

    return Layer(call, ())
end

function QuickGeLU()::Layer
    function call(x::Tensor)
        return Jacobi.quick_gelu(x)
    end

    return Layer(call, ())
end

function ELU()::Layer
    function call(x::Tensor)
        return Jacobi.elu(x)
    end

    return Layer(call, ())
end

function ReLU6()::Layer
    function call(x::Tensor)
        return Jacobi.relu6(x)
    end

    return Layer(call, ())
end

function HardSiLU()::Layer
    function call(x::Tensor)
        return Jacobi.hard_silu(x)
    end

    return Layer(call, ())
end

function LeakyReLU(α::Number = 0.01)::Layer
    function call(x::Tensor)
        return Jacobi.leaky_relu(x, α=α)
    end

    return Layer(call, ())
end

function GELU()::Layer
    function call(x::Tensor)
        return Jacobi.hard_silu(x)
    end

    return Layer(call, ())
end

function Softplus(β::Number = 1.0)::Layer
    function call(x::Tensor)
        return Jacobi.softplus(x, β=β)
    end

    return Layer(call, ())
end

function Mish()::Layer
    function call(x::Tensor)
        return Jacobi.mish(x)
    end

    return Layer(call, ())
end

function Softsign()::Layer
    function call(x::Tensor)
        return Jacobi.softsign(x)
    end

    return Layer(call, ())
end