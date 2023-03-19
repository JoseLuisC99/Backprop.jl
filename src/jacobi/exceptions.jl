import Base: showerror, Exception, @assert

macro assert(ex, err)
    esc(:($ex ? nothing : throw($err())))
end

mutable struct CyclicGraphException <: Exception end
Base.showerror(io::IO, e::CyclicGraphException) = print(io, "Computational graph contains loops")

mutable struct NoGradException <: Exception end
Base.showerror(io::IO, e::NoGradException) = print(io, "Tensor variable doesn't require gradient")

mutable struct NullGradException <: Exception end
Base.showerror(io::IO, e::NullGradException) = print(io, "Null gradient on tensor variable")

mutable struct MismatchDimsException <: Exception end
Base.showerror(io::IO, e::MismatchDimsException) = print(io, "Gradient dimension must match tensor dimension")