module Jacobi

export AbstractTensor, AutogradMetadata, Tensor, ones, zeros, eye, 
    arange, rand, randn, randexp, backward, zero_grad, clear_grads
export ⊙, multiply_n, reciprocal, relu, clamp, sigmoid, elu, silu,
    relu6, hard_silu, gelu, quick_gelu, leaky_relu, mish, softplus,
    softsign

include("tensor.jl")
include("math.jl")

@inline multiply_n(x::Tensor, y::Tensor)  = x ⊙ y

end