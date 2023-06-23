### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 9ff6db90-11f1-11ee-1f2d-f14e4c31aaa8
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	import Backprop
	import Backprop: Jacobi
	import Backprop.Jacobi: Tensor as Tensor

	import Backprop.nn as nn
end

# ╔═╡ afabca05-921f-4a7b-b0e7-da8a34f2ea24
model = nn.Sequential(
	nn.Linear(5, 6),
	nn.ReLU()
)

# ╔═╡ 4a037bbd-2865-4563-8cfc-91b40e15456d
begin
	x = Jacobi.randn((1, 5), requires_grad = true)
	o = model.forward(x)
end

# ╔═╡ 15aa7deb-ff00-4592-97a0-6b79cc3fad90
Jacobi.backward(o)

# ╔═╡ 517cf79b-3919-4aa8-9e17-f252dc589cf1
x

# ╔═╡ eb68eb42-ca0a-4bfe-9c62-c535c06e9323
Backprop.plot_tensor(o)

# ╔═╡ Cell order:
# ╠═9ff6db90-11f1-11ee-1f2d-f14e4c31aaa8
# ╠═afabca05-921f-4a7b-b0e7-da8a34f2ea24
# ╠═4a037bbd-2865-4563-8cfc-91b40e15456d
# ╠═15aa7deb-ff00-4592-97a0-6b79cc3fad90
# ╠═517cf79b-3919-4aa8-9e17-f252dc589cf1
# ╠═eb68eb42-ca0a-4bfe-9c62-c535c06e9323
