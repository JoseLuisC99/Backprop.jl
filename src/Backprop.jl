module Backprop

export Jacobi
export plot_tensor

include(joinpath("jacobi", "Jacobi.jl"))
include(joinpath("nn", "nn.jl"))
include("utils.jl")

end # module Backprop
