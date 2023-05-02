import Base: +, -, *, ^

tf_transpose = TensorFunction(
    "transpose",
    (x::Tensor) -> transpose(x.data),
    (x::Tensor) -> (g::Tensor) -> transpose(g)
)

tf_minus = TensorFunction(
    "-",
    (x::Tensor) -> -x.data,
    (x::Tensor) -> (g::Tensor) -> -g
)

tf_sum = TensorFunction(
    "+",
    (x::Tensor, y::Tensor) -> x.data + y.data,
    (x::Tensor, y::Tensor) -> (g::Tensor) -> (g, g)
)

tf_subtract = TensorFunction(
    "-",
    (x::Tensor, y::Tensor) -> x.data - y.data,
    (x::Tensor, y::Tensor) -> (g::Tensor) -> (g, -g)
)

tf_mult = TensorFunction(
    "*",
    (x::Tensor, y::Tensor) -> x.data * y.data,
    (x::Tensor, y::Tensor) -> (g::Tensor) -> (g * transpose(y), transpose(x) * g),
)

tf_hadamard_prod = TensorFunction(
    "⊙",
    (x::Tensor, y::Tensor) -> x.data .* y.data,
    (x::Tensor, y::Tensor) -> (g::Tensor) -> (Tensor(g.data .* y.data), Tensor(x.data .* g.data)),
)

tf_pow = TensorFunction(
    "^",
    (x::Tensor, n::Number) -> x.data .^ n,
    (x::Tensor, n::Number) -> (g::Tensor) -> g ⊙ Tensor(n .* x.data .^ (n - 1)),
)

tf_exp = TensorFunction(
    "exp",
    (x::Tensor) -> exp.(x.data),
    (x::Tensor) -> (g::Tensor) -> x ⊙ g
)

tf_logb = TensorFunction(
    "log",
    (b::Integer, x::Tensor) -> log.(b, x.data),
    (b::Integer, x::Tensor) -> (g::Tensor) -> reciprocal(x ⊙ fill(log(b), size(x))) ⊙ g
)

tf_log = TensorFunction(
    "log",
    (x::Tensor) -> log.(x.data),
    (x::Tensor) -> (g::Tensor) -> reciprocal(x) ⊙ g
)

tf_log2 = TensorFunction(
    "log2",
    (x::Tensor) -> log2.(x.data),
    (x::Tensor) -> (g::Tensor) -> reciprocal(x ⊙ fill(log(2), size(x))) ⊙ g
)

tf_log10 = TensorFunction(
    "log10",
    (x::Tensor) -> log10.(x.data),
    (x::Tensor) -> (g::Tensor) -> reciprocal(x ⊙ fill(log(10), size(x))) ⊙ g
)

tf_relu = TensorFunction(
    "relu",
    (x::Tensor) -> max.(x.data, 0),
    (x::Tensor) -> (g::Tensor) -> Tensor(1.0 * (x.data .> 0)) ⊙ g
)

tf_sin = TensorFunction(
    "sin",
    (x::Tensor) -> sin.(x.data),
    (x::Tensor) -> (g::Tensor) -> cos(x) ⊙ g
)

tf_cos = TensorFunction(
    "cos",
    (x::Tensor) -> cos.(x.data),
    (x::Tensor) -> (g::Tensor) -> -sin(x) ⊙ g
)

tf_tan = TensorFunction(
    "tan",
    (x::Tensor) -> tan.(x.data),
    (x::Tensor) -> (g::Tensor) -> Tensor(sec.(x.data) ^ 2) ⊙ g
)

tf_asin = TensorFunction(
    "asin",
    (x::Tensor) -> asin.(x.data),
    (x::Tensor) -> (g::Tensor) -> reciprocal(sqrt(ones_like(x) - x ^ 2)) ⊙ g
)

tf_acos = TensorFunction(
    "acos",
    (x::Tensor) -> acos.(x.data),
    (x::Tensor) -> (g::Tensor) -> -reciprocal(sqrt(ones_like(x) - x ^ 2)) ⊙ g
)

tf_atan = TensorFunction(
    "atan",
    (x::Tensor) -> atan.(x.data),
    (x::Tensor) -> (g::Tensor) -> reciprocal(ones(size(x) + x ^ 2)) ⊙ g
)

tf_sinh = TensorFunction(
    "sinh",
    (x::Tensor) -> sinh.(x.data),
    (x::Tensor) -> (g::Tensor) -> cosh(x) ⊙ g
)

tf_cosh = TensorFunction(
    "cosh",
    (x::Tensor) -> cosh.(x.data),
    (x::Tensor) -> (g::Tensor) -> sinh(x) ⊙ g
)

tf_tanh = TensorFunction(
    "tanh",
    (x::Tensor) -> tanh.(x.data),
    (x::Tensor) -> (g::Tensor) -> (ones_like(x) - tanh(x)) ⊙ g
)

Base.transpose(x::Tensor) = apply!(tf_transpose, x)
Base.:-(x::Tensor) = apply!(tf_minus, x)
Base.:+(x::Tensor, y::Tensor) = apply!(tf_sum, x, y)
Base.:-(x::Tensor, y::Tensor) = apply!(tf_subtract, x, y)
Base.:*(x::Tensor, y::Tensor) = apply!(tf_mult, x, y)
⊙(x::Tensor, y::Tensor) = apply!(tf_hadamard_prod, x, y)

Base.:^(x::Tensor, n::Number) = apply!(tf_pow, x, n)
Base.sqrt(x::Tensor) = x ^ 0.5
reciprocal(x::Tensor) = Base.:^(x, -1)

Base.exp(x::Tensor) = apply!(tf_exp, x)
Base.log(b::Integer, x::Tensor) = apply!(tf_logb, b, x)
Base.log(x::Tensor) = apply!(tf_log, x)
Base.log2(x::Tensor) = apply!(tf_log2, x)
Base.log10(x::Tensor) = apply!(tf_log10, x)

relu(x::Tensor) = apply!(tf_relu, x)
Base.abs(x::Tensor) = relu(x) + relu(-x)
Base.sign(x::Tensor) = x ⊙ reciprocal(abs(x) + fill(eps(), size(x)))
function clamp(x::Tensor, min::Number, max::Number)::Tensor
    min_ = fill(min, size(x))
    max_ = fill(max, size(x))
    return relu(x - min_) + min_ - relu(x - max_)
end

Base.sin(x::Tensor) = apply!(tf_sin, x)
Base.cos(x::Tensor) = apply!(tf_cos, x)
Base.tan(x::Tensor) = apply!(tf_tan, x)

Base.asin(x::Tensor) = apply!(tf_asin, x)
Base.acos(x::Tensor) = apply!(tf_acos, x)
Base.atan(x::Tensor) = apply!(tf_atan, x)

Base.sinh(x::Tensor) = apply!(tf_sinh, x)
Base.cosh(x::Tensor) = apply!(tf_cosh, x)
Base.tanh(x::Tensor) = apply!(tf_tanh, x)

@inline sigmoid(x::Tensor) = reciprocal(ones_like(x) + exp(-x))
@inline elu(x::Tensor; α::Number = 1.0) = relu(x) - fill(α, size(x)) ⊙ relu(ones_like(x) - exp(x))
@inline silu(x::Tensor) = x ⊙ sigmoid(x)
@inline relu6(x::Tensor) = relu(x) - relu(x - fill(6, size(x)))
@inline hard_silu(x::Tensor) = x ⊙ relu6(x + fill(3, size(x))) ⊙ fill(1/6, size(x))
@inline gelu(x::Tensor) = fill(0.5, size(x)) ⊙ (ones_like(x) + tanh(x ⊙ fill(0.7978845608, size(x)) ⊙ (ones_like(x) + fill(0.044715, size(x)) ⊙ (x ^ 2))))
@inline quick_gelu(x::Tensor) = x ⊙ sigmoid(x ⊙ fill(1.702, size(x)))
@inline leaky_relu(x::Tensor; α::Number = 0.01) = relu(x) - relu(-fill(α, size(x)) ⊙ x)
@inline mish(x::Tensor) = x ⊙ tanh(softplus(x))
@inline softplus(x::Tensor; β::Number = 1.0) = fill(1 / β, size(x)) ⊙ log(ones_like(x) + exp(x ⊙ fill(β, size(x))))
@inline softsign(x::Tensor) = x ⊙ reciprocal(ones_like(x) + abs(x))

# function Base.maximum(x::Tensor)::Number
# end

# function Base.minimum(x::Tensor)::Number
#     return -maximum(-x)
# end

# function Base.reduce(op::Function, x::Tensor)::Number
# end