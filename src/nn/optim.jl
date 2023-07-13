import ..Jacobi
import ..Jacobi: Tensor as Tensor
import ..Jacobi: ⊙ as ⊙

struct Optimizer
    step::Function
    params::Set{<:Tensor}
end

function GradientDescent(params::Set{<:Tensor}; lr::Float64 = 0.001)
    function step(x::Tensor)
        return x - (Tensor(lr) ⊙ x.grad)
    end

    return Optimizer(step, params)
end

function step(optim::Optimizer)::Nothing
    for param in optim.params
        d = optim.step(param)
        param.data = d.data
    end
end