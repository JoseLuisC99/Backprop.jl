import ..Jacobi
import ..Jacobi: Tensor as Tensor
import ..Jacobi: ⊙ as ⊙

@enum Reduction Mean = 0 Sum

function apply_reduction(x::Tensor, reduction::Reduction)::Tensor
    if reduction == Mean::Reduction
        return Jacobi.mean(x)
    elseif reduction == Sum::Reduction
        return sum(x)
    end
end

function l1_loss(p::Tensor, y::Tensor; reduction::Reduction = Mean::Reduction)::Tensor
    l = abs(p - y)
    return apply_reduction(l, reduction)
end

function mse_loss(p::Tensor, y::Tensor; reduction::Reduction = Mean::Reduction)::Tensor
    l = (p - y) ^ 2
    return apply_reduction(l, reduction)
end

function bce_loss(p::Tensor, y::Tensor; from_logits::Bool = true, reduction::Reduction = Mean::Reduction)::Tensor
    if from_logits
        p = Jacobi.sigmoid(p)
    end
    l = (y ⊙ log(Tensor(eps()) .+ p)) + ((Tensor(1) .- y) ⊙ log(Tensor(1 + eps()) .- p))
    return apply_reduction(-l, reduction)
end

# function crossentropy_loss(x::Tensor, y::Tensor; from_logits::Bool = true, reduction::Reduction = Mean::Reduction)::Tensor
#     if from_logits
#         x = Jacobi.softmax(x)
#     end
#     l = (y ⊙ log(x)) + ((Tensor(1) .+ y) ⊙ log(Tensor(1) .- x))
#     return -apply_reduction(l, reduction)
# end