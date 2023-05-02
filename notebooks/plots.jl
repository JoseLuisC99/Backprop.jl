### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 18460162-e6ba-11ed-2236-a52c3773342e
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	import Backprop
	import Backprop: Jacobi
	import Backprop.Jacobi: Tensor as Tensor
end

# ╔═╡ fe5b62d6-4090-462d-b4f8-d1a8bffedae1
begin
	x = Jacobi.rand(3, name="x", requires_grad=true)
	b = Jacobi.rand(3, name="b", requires_grad=true)
	W = Jacobi.rand((3, 3), name="W", requires_grad=true)

	t = log(Jacobi.relu(W * x + b) + x)
end

# ╔═╡ 847508de-f312-49c4-83c9-8a6ed7bfadec
Backprop.plot_tensor(t)

# ╔═╡ e5325aa4-74ac-45a0-b666-9c524101a851
Jacobi.backward(t)

# ╔═╡ 4e9f1738-6e22-412b-a9c7-521f42e2279c
Backprop.plot_tensor(x.grad, rankdir="BT")

# ╔═╡ 10f9a984-435f-4a3a-b731-54e5f58e4be3
Backprop.plot_tensor(W.grad, rankdir="BT")

# ╔═╡ Cell order:
# ╠═18460162-e6ba-11ed-2236-a52c3773342e
# ╠═fe5b62d6-4090-462d-b4f8-d1a8bffedae1
# ╠═847508de-f312-49c4-83c9-8a6ed7bfadec
# ╠═e5325aa4-74ac-45a0-b666-9c524101a851
# ╠═4e9f1738-6e22-412b-a9c7-521f42e2279c
# ╠═10f9a984-435f-4a3a-b731-54e5f58e4be3
