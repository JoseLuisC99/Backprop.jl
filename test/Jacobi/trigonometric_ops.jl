using Test
import Backprop: Jacobi
import Backprop.Jacobi: Tensor as Tensor, ⊙ as ⊙

@testset "sin, cos and tan" begin
    a = Tensor([2.6679  0.3260 -0.0724  1.8656;
         0.0683  0.6037  1.3719 -0.6429;
         0.5886  0.6979  0.8043 -0.0526;
        -0.8386  0.4058 -0.1800 -1.0571;
        -0.9748 -1.3169  0.5479 -0.9574], requires_grad=true)
    b = Tensor([-0.3685  0.0211 -0.7277;
        -1.4396  1.5052 -1.0041;
         1.0104  0.0937 -0.6065;
         0.4545  0.8518 -1.3036], requires_grad=false)
    c = Tensor([0.1701 -0.0142  0.0749;
        -1.5749 -0.2265 -0.8595;
        -0.0821 -0.7563 -0.3989;
         0.1670  1.3138  1.6082;
        -0.6658  0.6985 -0.2831], requires_grad=true)
    
    t = sin(a * b + c)
    Jacobi.backward(t)
    @test t ≈ Tensor([-0.4861  0.8555  0.9915;
        -0.9809  0.2615 -0.9981;
        -0.4925  0.3308 -0.9300;
        -0.6965  0.8358 -0.1560;
         0.9906 -0.8787  0.4594]) atol=1e-3 
    @test a.grad ≈ Tensor([-0.2383 -1.9070  0.9134  0.1257;
        -0.0959  1.1114  0.2496  0.8305;
        -0.0333  0.5366  1.1907  1.6786;
         0.4660  0.7852  1.3756  2.0814;
         0.6866  0.3698  0.3561  0.6893]) atol=1e-3
    @test b.grad === nothing
    @test c.grad ≈ Tensor([0.8739 -0.5178 -0.1301;
         0.1944  0.9652  0.0614;
         0.8703  0.9437 -0.3676;
         0.7176  0.5490 -0.9878;
        -0.1365 -0.4774 -0.8883]) atol=1e-3
    Jacobi.zero_grad(t)

    t = cos(a * b + c)
    Jacobi.backward(t)
    @test t ≈ Tensor([0.8739 -0.5178 -0.1301;
         0.1944  0.9652  0.0614;
         0.8703  0.9437 -0.3676;
         0.7176  0.5490 -0.9878;
        -0.1365 -0.4774 -0.8883]) atol=1e-3
    @test a.grad ≈ Tensor([0.524357 -0.991816  1.012299  0.784743;
        -1.093316 -2.807930  0.361275 -1.078036;
        -0.865208 -2.140700 -0.097429 -1.270274;
        -0.387782 -2.417326  0.530825 -0.598701;
         0.717864  3.210002 -0.640016  0.897048]) atol=1e-3
    @test b.grad === nothing
    @test c.grad ≈ Tensor([0.486054 -0.855477 -0.991505;
         0.980929 -0.261480  0.998113;
         0.492481 -0.330805  0.929984;
         0.696487 -0.835813  0.155959;
        -0.990646  0.878709 -0.459352]) atol=1e-3
    Jacobi.zero_grad(t)

    t = tan(a * b + c)
    Jacobi.backward(t)
    @test t ≈ Tensor([-0.5561709 -1.6520051 -7.6230712;
        -5.0468173 0.2709050 -16.2538624;
        -0.5658595 0.3505411   2.5298734;
        -0.9706188 1.5223882   0.1578912;
        -7.2596288 1.8407776  -0.5171407]) atol=1e-3
    @test a.grad ≈ Tensor([-43.4190292 -55.6254044 -34.1785889 -73.2858200;
        -202.7090149 -302.7663574 -133.9903259 -332.7540283;
          -5.8479691   -7.6409984   -3.0491166   -8.0904808;
          -1.3915027    1.1687707    1.6515441    2.3725743;
         -20.6189804  -71.9768143   53.9032135   26.4935188]) atol=1e-3
    @test b.grad === nothing
    @test c.grad ≈ Tensor([1.3093261 3.7291207 59.1112137;
        26.4703655 1.0733895 265.1880493;
         1.3201970 1.1228790   7.4002595;
         1.9421008 3.3176658   1.0249296;
        53.7022095 4.3884621   1.2674346]) atol=1e-3
    Jacobi.zero_grad(t)
end

@testset "sinh, cosh and tanh" begin
    a = Tensor([2.6679  0.3260 -0.0724  1.8656;
         0.0683  0.6037  1.3719 -0.6429;
         0.5886  0.6979  0.8043 -0.0526;
        -0.8386  0.4058 -0.1800 -1.0571;
        -0.9748 -1.3169  0.5479 -0.9574], requires_grad=true)
    b = Tensor([-0.3685  0.0211 -0.7277;
        -1.4396  1.5052 -1.0041;
         1.0104  0.0937 -0.6065;
         0.4545  0.8518 -1.3036], requires_grad=false)
    c = Tensor([0.1701 -0.0142  0.0749;
        -1.5749 -0.2265 -0.8595;
        -0.0821 -0.7563 -0.3989;
         0.1670  1.3138  1.6082;
        -0.6658  0.6985 -0.2831], requires_grad=true)
    
    t = sinh(a * b + c)
    Jacobi.backward(t)
    @test t ≈ Tensor([-0.529645 4.084990 -48.847397;
        -1.851509  0.267652 -2.151368;
        -0.537998  0.343581 -3.433268;
        -0.849019  1.159235 13.513345;
         2.667440 -3.893049  7.144316]) atol=1e-3 
    @test a.grad ≈ Tensor([-35.881958 -44.356720 -28.094723 -59.594162;
         -2.480003  -3.853318  0.784311  -1.254501;
         -2.998344  -3.633751 -0.922384  -3.244816;
        -10.311646 -13.189930 -6.749355 -15.763878;
         -6.214545  -5.294516 -1.120295  -4.685623]) atol=1e-3
    @test b.grad === nothing
    @test c.grad ≈ Tensor([1.131602 4.205608 48.857635;
        2.104301 1.035199  2.372421;
        1.135536 1.057378  3.575937;
        1.311805 1.530956 13.550294;
        2.848726 4.019432  7.213962]) atol=1e-3
    Jacobi.zero_grad(t)

    t = cosh(a * b + c)
    Jacobi.backward(t)
    @test t ≈ Tensor([1.131602 4.205608 48.857635;
        2.104301 1.035199  2.372421;
        1.135536 1.057378  3.575937;
        1.311805 1.530956 13.550294;
        2.848726 4.019432  7.213962]) atol=1e-3
    @test a.grad ≈ Tensor([35.827618 55.958870 29.473558 66.916336;
         2.253479   5.228490 -0.540881   2.190998;
         2.703891   4.739004  1.570877   4.523749;
        -9.496338 -10.601622 -8.945072 -17.014439;
        -6.264014 -16.873474 -2.002625 -11.417078]) atol=1e-3
    @test b.grad === nothing
    @test c.grad ≈ Tensor([-0.529645 4.084990 -48.847397;
        -1.851509  0.267652 -2.151368;
        -0.537998  0.343581 -3.433268;
        -0.849019  1.159235 13.513345;
         2.667440 -3.893049  7.144316]) atol=1e-3
    Jacobi.zero_grad(t)

    t = tanh(a * b + c)
    Jacobi.backward(t)
    @test t ≈ Tensor([-0.4680  0.9713 -0.9998;
        -0.8799  0.2586 -0.9068;
        -0.4738  0.3249 -0.9601;
        -0.6472  0.7572  0.9973;
         0.9364 -0.9686  0.9903]) atol=1e-3
    @test a.grad ≈ Tensor([-0.2869 -1.0395  0.7941  0.4025;
        -0.1928  0.9011 0.2079 0.6659;
        -0.3238  0.1513 0.8200 1.0124;
        -0.2091 -0.1998 0.6238 0.6204;
        -0.0581 -0.1035 0.1187 0.0837]) atol=1e-3
    @test b.grad === nothing
    @test c.grad ≈ Tensor([7.8093e-01 5.6538e-02 4.1898e-04;
        2.2583e-01 9.3315e-01 1.7767e-01;
        7.7553e-01 8.9442e-01 7.8202e-02;
        5.8111e-01 4.2665e-01 5.4463e-03;
        1.2323e-01 6.1897e-02 1.9215e-02]) atol=1e-3
    Jacobi.zero_grad(t)
end