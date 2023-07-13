import ..Jacobi
import ..Jacobi: Tensor as Tensor
import Random

function make_moons(n_samples::Int64, noise::Float64 = 0.2)
	split = n_samples ÷ 2
	X = Array{Float64}(undef, n_samples, 2)
	
	x1 = 2 .* Random.rand(split)
	x2 = sqrt.(1 .- (x1 .- 1).^ 2) + (noise .* Random.randn(split)) .- 1
	X[begin:split, 1] = x1
	X[begin:split, 2] = x2

	x1 = 1 .+ 2 .* Random.rand(split)
	x2 = -(sqrt.(1 .- (x1 .- 2).^ 2) + (noise .* Random.randn(split)))
	X[split + 1:end, 1] = x1
	X[split + 1:end, 2] = x2

	y = [zeros(Int64, 500); ones(Int64, 500)]
	idx = Random.randperm(1000)
	
	return Tensor(X[idx, :]), Tensor(y[idx])
end