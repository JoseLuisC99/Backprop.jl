module nn

export AbstractLayer, Model, Sequential
export Layer, Linear, ReLU

include("layers.jl")
include("model.jl")

end