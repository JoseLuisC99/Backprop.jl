### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 3382df90-c741-11ed-15a2-856666217a49
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	import Backprop: Jacobi
	import Backprop.Jacobi: Tensor as Tensor
end

# ╔═╡ 53398b38-5f83-4b56-9a36-f183de8c0a1a
begin
	x = Jacobi.rand((5, 4), requires_grad=true)
	y = Jacobi.rand((5, 4), requires_grad=true)
end

# ╔═╡ 127d868f-ebca-468d-a1a1-6fe2363e38ad
begin
	a = Tensor([0.7666 0.4962 0.2044 0.5679;
		0.1905 0.1287 0.5959 0.0406;
		0.5544 0.6452 0.6137 0.5141;
		0.8700 0.1873 0.6246 0.4608], requires_grad=true)
	b = Tensor([0.2305 0.2752 0.9448 0.1839;
		0.9254 0.5193 0.8589 0.6293;
		0.0951 0.6552 0.4011 0.9452;
		0.9161 0.2019 0.1014 0.4723], requires_grad=true)
	t = log10(a) + log10(b)
	Jacobi.backward(t)
end

# ╔═╡ 8a43eeab-1418-4eeb-83a1-615c7473daa4
q = tanh(a) + b

# ╔═╡ a8a60c9d-156a-46db-b2d1-f5a77612ec64
Jacobi.backward(q)

# ╔═╡ 3b0d24b1-4f1d-418a-9235-617b6fdaff37


# ╔═╡ Cell order:
# ╠═3382df90-c741-11ed-15a2-856666217a49
# ╠═53398b38-5f83-4b56-9a36-f183de8c0a1a
# ╠═127d868f-ebca-468d-a1a1-6fe2363e38ad
# ╠═8a43eeab-1418-4eeb-83a1-615c7473daa4
# ╠═a8a60c9d-156a-46db-b2d1-f5a77612ec64
# ╠═3b0d24b1-4f1d-418a-9235-617b6fdaff37
