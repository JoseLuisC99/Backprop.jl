### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 05ba7580-1eaf-11ee-2780-bf4749a5c3b2
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	import Backprop
	import Backprop: Jacobi
	import Backprop.Jacobi: Tensor as Tensor
	import Backprop.Jacobi: ⊙ as ⊙
	import Backprop.nn as nn
	import Backprop.data as data 

	using Plots
	import Random
end

# ╔═╡ 49830e50-6964-4e27-b196-de56bda63d26
begin
	X, y = data.make_moons(1000)
	dataset = data.Dataset(X, y, 64)

	scatter(X[:, 1][y .== 0].data, X[:, 2][y .== 0].data, label="Class 0")
	scatter!(X[:, 1][y .== 1].data, X[:, 2][y .== 1].data, label="Class 1")
end

# ╔═╡ 2ce16a18-445e-4592-ba15-381986b3b228
begin 
	model = nn.Sequential(
		nn.Linear(2,  8),
		nn.Tanh(),
		nn.Linear(8, 16),
		nn.Tanh(),
		nn.Linear(16, 8),
		nn.Tanh(),
		nn.Linear(8, 1),
		nn.Sigmoid(),
	)
end

# ╔═╡ 0e9eb7f3-011b-4469-b666-aafb46328e3a
function train(model::nn.Model, dataset::data.Dataset; α::Float64 = 0.1, epochs::Int64 = 200)
	history = Vector{Float64}()
	optimizer = nn.GradientDescent(model.params, lr=α)

	anim = @animate for _ in 1:epochs
		Xr = 0:0.1:3
		Yr = -1.5:0.1:0.5
		function boundary(x, y)
			p = model.forward(Tensor([x y]))
			return p.data[1]
		end
		scatter(X[:, 1][y .== 0].data, X[:, 2][y .== 0].data, legend = false)
		scatter!(X[:, 1][y .== 1].data, X[:, 2][y .== 1].data, legend = false)
		contourf!(Xr, Yr, boundary, levels=5, color=:turbo, fillalpha=0.2, lw=0)
		
		epoch_loss = 0.0
		for (Xi, yi) in dataset
			pi = model.forward(Xi)
			loss = nn.bce_loss(pi, yi)
			epoch_loss += loss.data[1]
			
			Jacobi.backward!(loss)
			nn.step(optimizer)
			Jacobi.clear_grads(loss)
		end
		push!(history, epoch_loss / length(dataset))
	end

	return history, anim
end

# ╔═╡ c072e187-0051-4d30-bd3b-6687a9ba68be
history, anim = train(model, dataset)

# ╔═╡ 81de29b2-4afe-4869-a0d1-cd9412485a8e
plot(history, label="Training loss")

# ╔═╡ a9ab1fa3-9577-43ae-aa44-d49a1be19c92
gif(anim, "Animation.gif", fps = 15)

# ╔═╡ Cell order:
# ╠═05ba7580-1eaf-11ee-2780-bf4749a5c3b2
# ╠═49830e50-6964-4e27-b196-de56bda63d26
# ╠═2ce16a18-445e-4592-ba15-381986b3b228
# ╠═0e9eb7f3-011b-4469-b666-aafb46328e3a
# ╠═c072e187-0051-4d30-bd3b-6687a9ba68be
# ╠═81de29b2-4afe-4869-a0d1-cd9412485a8e
# ╠═a9ab1fa3-9577-43ae-aa44-d49a1be19c92
