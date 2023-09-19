using Test
import Backprop: Jacobi
import Backprop.Jacobi: Tensor as Tensor, ⊙ as ⊙

@testset "Forward execution algebra" begin
    m = Tensor([4.4, 0.9, 2.6])
    n = Tensor([0.2, 1.9, 7.5])
    p = Tensor([1.9, 7.8, 5.6])
    q = Tensor([1.3, 0.4, 2.9, 2.2])

    @test m + n ≈ Tensor([4.6, 2.8, 10.1])
    @test m + n + p ≈ Tensor([6.5, 10.6, 15.7])
    @test (m ⊙ n) + p ≈ Tensor([2.78, 9.51, 25.1])
    @test (m - n) + p ≈ Tensor([6.1, 6.8, 0.7])
    @test -(p ⊙ n) ≈ Tensor([-0.38, -14.82, -42.0])
    @test (m + n) ^ 2 ≈ Tensor([21.16, 7.84, 102.01])
    @test (m + n) ^ 0.5 ≈ Tensor([2.1448, 1.6733, 3.178]) atol=1e-4
    @test_throws DimensionMismatch m + q
    @test_throws DimensionMismatch m - q
    @test_throws DimensionMismatch m ⊙ q

    W = Tensor([-1.2844 -3.0717 -0.0539  0.6434;
                -0.6646 -0.4634 -0.1146  0.5935;
                 1.8421 -1.0111  0.6293  0.2473;
                 0.2080  0.1901  1.4661  0.5014;
                 1.7475  0.1142  0.3445 -0.7199])
    x = Tensor([0.1503, -0.9509, 1.7091, 1.0097])
    b = Tensor([-0.0478, 1.7576, -0.0732, -0.1256, -2.4584])
    V = Tensor([-1.9776 -1.5477 -0.2004;
                -0.0098  0.2112 -0.4579;
                 2.2112  0.7707 -0.0945;
                -0.0332  0.1591 -0.3489])
    U = Tensor([-0.6027  0.1165  1.4423;
                -0.9416  1.0732 -0.5358;
                -0.2921 -0.8288 -1.7697;
                -0.8759 -1.1469 -1.0327;
                -2.5232 -0.6087 -0.6090])
    Y = Tensor([-1.2500 -1.1406  0.9118;
                -4.1188 -0.7481  0.5305;
                -0.7784 -2.0328 -5.7128;
                -1.4403 -2.0062 -8.1338;
                -3.9648  0.3382  2.6229])

    @test W * x ≈ Tensor([3.2854, 0.7442, 2.5636, 2.8625, 0.0160]) atol=1e-4
    @test W * x + b ≈ Tensor([ 3.2376, 2.5018, 2.4904, 2.7369, -2.4424]) atol=1e-4
    @test W * V ≈ Tensor([ 2.4296  1.3999  1.4445;
         1.0457  0.9368  0.1491;
        -2.2497 -2.5402 -0.0519;
         2.8120  0.9279 -0.4422;
        -2.6713 -2.5295 -0.1839]) atol=1e-3
    @test (W * V) + U ≈ Tensor([1.8269  1.5164  2.8868;
         0.1041  2.0100 -0.3867;
        -2.5418 -3.3690 -1.8216;
         1.9361 -0.2190 -1.4749;
        -5.1945 -3.1382 -0.7929]) atol=1e-3
    @test (U + Y) ^ 2 ≈ Tensor([3.4325e+00 1.0488e+00 5.5418e+00;
        2.5608e+01 1.0569e-01 2.8090e-05;
        1.1460e+00 8.1888e+00 5.5988e+01;
        5.3648e+00 9.9420e+00 8.4025e+01;
        4.2094e+01 7.3170e-02 4.0558e+00]) atol=1e-3
    @test U ⊙ Y ≈ Tensor([0.7534 -0.1329  1.3151;
        3.8783 -0.8029 -0.2842;
        0.2274  1.6848 10.1099;
        1.2616  2.3009  8.3998;
        10.0040 -0.2059 -1.5973]) atol=1e-3
    @test_throws DimensionMismatch W * x + m
    @test_throws DimensionMismatch W * n
    @test_throws DimensionMismatch W * U
    @test_throws DimensionMismatch W + U
end

@testset "Backward execution algebra" begin
    m = Tensor([4.4, 0.9, 2.6], requires_grad=true)
    n = Tensor([0.2, 1.9, 7.5], requires_grad=true)
    p = Tensor([1.9, 7.8, 5.6])
    q = Tensor([1.3, 0.4, 2.9, 2.2])

    t = m + n + p
    Jacobi.backward!(t)
    @test m.grad ≈ Jacobi.ones(size(m))
    @test n.grad ≈ Jacobi.ones(size(n))
    @test p.grad === nothing
    Jacobi.zero_grad(t)
    @test m.grad ≈ Jacobi.zeros(size(m))
    @test n.grad ≈ Jacobi.zeros(size(n))
    @test p.grad === nothing

    t = (m ⊙ n) + p
    Jacobi.backward!(t)
    @test m.grad ≈ Tensor([0.2, 1.9, 7.5])
    @test n.grad ≈ Tensor([4.4, 0.9, 2.6])
    @test p.grad === nothing
    Jacobi.zero_grad(t)

    t = (m - n) + p
    Jacobi.backward!(t)
    @test m.grad ≈ Jacobi.ones(size(m))
    @test n.grad ≈ -Jacobi.ones(size(n))
    @test p.grad === nothing
    Jacobi.zero_grad(t)

    t = -(p ⊙ n)
    Jacobi.backward!(t)
    @test n.grad ≈ Tensor([-1.9, -7.8, -5.6])
    @test p.grad === nothing
    Jacobi.zero_grad(t)

    t = (m + n) ^ 2
    Jacobi.backward!(t)
    @test m.grad ≈ Tensor([9.2, 5.6, 20.2]) atol=1e-3
    @test n.grad ≈ Tensor([9.2, 5.6, 20.2]) atol=1e-3
    Jacobi.zero_grad(t)

    t = (m + n) ^ 0.5
    Jacobi.backward!(t)
    @test m.grad ≈ Tensor([0.2331, 0.2988, 0.1573]) atol=1e-3
    @test n.grad ≈ Tensor([0.2331, 0.2988, 0.1573]) atol=1e-3
    Jacobi.zero_grad(t)

    W = Tensor([-1.2844 -3.0717 -0.0539  0.6434;
                -0.6646 -0.4634 -0.1146  0.5935;
                 1.8421 -1.0111  0.6293  0.2473;
                 0.2080  0.1901  1.4661  0.5014;
                 1.7475  0.1142  0.3445 -0.7199], requires_grad=true)
    x = Tensor([0.1503, -0.9509, 1.7091, 1.0097])
    b = Tensor([-0.0478, 1.7576, -0.0732, -0.1256, -2.4584], requires_grad=true)
    V = Tensor([-1.9776 -1.5477 -0.2004;
                -0.0098  0.2112 -0.4579;
                 2.2112  0.7707 -0.0945;
                -0.0332  0.1591 -0.3489])
    U = Tensor([-0.6027  0.1165  1.4423;
                -0.9416  1.0732 -0.5358;
                -0.2921 -0.8288 -1.7697;
                -0.8759 -1.1469 -1.0327;
                -2.5232 -0.6087 -0.6090], requires_grad=true)
    Y = Tensor([-1.2500 -1.1406  0.9118;
                -4.1188 -0.7481  0.5305;
                -0.7784 -2.0328 -5.7128;
                -1.4403 -2.0062 -8.1338;
                -3.9648  0.3382  2.6229], requires_grad=true)

    t = W * x + b
    Jacobi.backward!(t)
    @test W.grad ≈ Tensor([0.1503 -0.9509  1.7091  1.0097;
        0.1503 -0.9509  1.7091  1.0097;
        0.1503 -0.9509  1.7091  1.0097;
        0.1503 -0.9509  1.7091  1.0097;
        0.1503 -0.9509  1.7091  1.0097]) atol=1e-3
    @test b.grad ≈ Jacobi.ones(size(b)) atol=1e-3
    @test x.grad === nothing
    Jacobi.zero_grad(t)

    t = (W * V) + U
    Jacobi.backward!(t)
    @test W.grad ≈ Tensor([-3.7257 -0.2565  2.8874 -0.2230;
        -3.7257 -0.2565 2.8874 -0.2230;
        -3.7257 -0.2565 2.8874 -0.2230;
        -3.7257 -0.2565 2.8874 -0.2230;
        -3.7257 -0.2565 2.8874 -0.2230]) atol=1e-3
    @test U.grad ≈ Jacobi.ones(size(U)) atol=1e-3
    @test V.grad === nothing
    Jacobi.zero_grad(t)

    t = (U + Y) ^ 2
    Jacobi.backward!(t)
    @test U.grad ≈ Tensor([-3.7054e+00 -2.0482e+00  4.7082e+00;
        -1.0121e+01  6.5020e-01 -1.0600e-02;
        -2.1410e+00 -5.7232e+00 -1.4965e+01;
        -4.6324e+00 -6.3062e+00 -1.8333e+01;
        -1.2976e+01 -5.4100e-01  4.0278e+00]) atol=1e-3
    @test Y.grad ≈ Tensor([-3.7054e+00 -2.0482e+00  4.7082e+00;
        -1.0121e+01  6.5020e-01 -1.0600e-02;
        -2.1410e+00 -5.7232e+00 -1.4965e+01;
        -4.6324e+00 -6.3062e+00 -1.8333e+01;
        -1.2976e+01 -5.4100e-01  4.0278e+00]) atol=1e-3
    Jacobi.zero_grad(t)

    t = U ⊙ Y
    Jacobi.backward!(t)
    @test U.grad ≈ Tensor([-1.2500 -1.1406  0.9118;
        -4.1188 -0.7481  0.5305;
        -0.7784 -2.0328 -5.7128;
        -1.4403 -2.0062 -8.1338;
        -3.9648  0.3382  2.6229]) atol=1e-3
    @test Y.grad ≈ Tensor([-0.6027 0.1165 1.4423;
        -0.9416  1.0732 -0.5358;
        -0.2921 -0.8288 -1.7697;
        -0.8759 -1.1469 -1.0327;
        -2.5232 -0.6087 -0.6090]) atol=1e-3
    Jacobi.zero_grad(t)
end