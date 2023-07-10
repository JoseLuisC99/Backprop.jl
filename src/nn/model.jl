import ..Jacobi
import ..Jacobi: Tensor as Tensor

mutable struct Model
    params::Set{<:Tensor}
    forward::Function
end

function Sequential(layers::Vararg{<:AbstractLayer})::Model
    function forward(x::Tensor)::Tensor
        for layer in layers
            x = layer.call(x)
        end
        return x
    end

    params = Set{Tensor}()
    for layer in layers
        if length(layer.params) > 0
            push!(params, layer.params...)
        end
    end

    return Model(params, forward)
end

function forward(model::Model)::Nothing
end