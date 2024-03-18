export MvNormalGamma

import BayesBase
using LinearAlgebra
using Distributions
using SpecialFunctions

struct MvNormalGamma <: ContinuousMultivariateDistribution
 
    D ::Integer
    μ ::Vector
    Λ ::Matrix
    α ::Real
    β ::Real

    function MvNormalGamma(mean::Vector, precision_matrix::Matrix, shape::Float64, rate::Float64)
        
        if shape <= 0.0; error("Shape parameter must be positive."); end
        if rate <= 0.0;  error("Rate parameter must be positive."); end
        
        dimensions = length(mean)
        if size(precision_matrix, 1) != dimensions
            error("Number of rows of precision matrix does not match mean vector length.")
        end
        if size(precision_matrix, 2) != dimensions
            error("Number of columns of precision matrix does not match mean vector length.")
        end

        return new(dimensions, mean, precision_matrix, shape, rate)
    end
end

BayesBase.dims(d::MvNormalGamma) = d.D
BayesBase.params(d::MvNormalGamma) = (d.μ, d.Λ, d.α, d.β)

function BayesBase.pdf(dist::MvNormalGamma, x::Vector)
    μ, Λ, α, β = params(dist)
    θ = x[1:end-1]
    τ = x[end]
    return det(Λ)^(1/2) * (2π)^(-p.D/2)*β^α/gamma(α)*τ^(α+p.D/2-1)*exp( -τ/2*((θ-μ)'*Λ*(θ-μ) +2β) )
end

function BayesBase.logpdf(dist::MvNormalGamma, x::Vector)
    μ, Λ, α, β = params(dist)
    θ = x[1:end-1]
    τ = x[end]
    return 1/2*logdet(Λ) -p.D/2*log(2π) + α*log(β) - log(gamma(α)) +(α+p.D/2-1)*log(τ) -τ/2*((θ-μ)'*Λ*(θ-μ) +2β)
end

BayesBase.default_prod_rule(::Type{<:MvNormalGamma}, ::Type{<:MvNormalGamma}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalGamma, right::MvNormalGamma)
    μl, Λl, αl, βl = params(left)
    μr, Λr, αr, βr = params(right)

    return MvNormalGamma(μ, Λ, α, β)
end