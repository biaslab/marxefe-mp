using LinearAlgebra

@testitem "MatrixNormalWishartProps" begin

    Dx = 3
    Dy = 2

    ν0 = 10
    M0 = zeros(Dx,Dy)
    Λ0 = diagm(ones(Dx))
    Ω0 = diagm(ones(Dy))
    p0 = MatrixNormalWishart(M0,Λ0,Ω0,ν0)

    A = zeros(Dx,Dy)
    W = diagm(ones(Dy))

    @testset "init" begin
        @test typeof(MatrixNormalWishart(M0,Λ0,Ω0,ν0)) <: MatrixNormalWishart
        @test_throws ErrorException MatrixNormalWishart(M0,Ω0,Λ0,ν0)
        @test_throws ErrorException MatrixNormalWishart(zeros(Dy,Dx),Ω0,Λ0,ν0)
        @test_throws MethodError MatrixNormalWishart(M0',Λ0,Ω0,ν0)
        @test_throws ErrorException MatrixNormalWishart(M0,Λ0,Ω0,-ν0)
    end

    @testset "pdfs" begin
        @test isapprox(pdf(MatrixNormalWishart(M0,Λ0,Ω0,ν0), (A,W)), 2.92708629009271e-9, atol=1e-10)
        @test isapprox(logpdf(MatrixNormalWishart(M0,Λ0,Ω0,ν0), (A,W)), -19.649258348942578, atol=1e-8)
        @test_throws MethodError pdf(MatrixNormalWishart(M0,Λ0,Ω0,ν0), A,W)
    end

    @testset "prod" begin
        @test typeof(prod(GenericProd(), p0,p0)) <: MatrixNormalWishart
    end

end