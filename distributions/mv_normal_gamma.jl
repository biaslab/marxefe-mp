export MvNormalGamma

import BayesBase
using LinearAlgebra
using Distributions
using SpecialFunctions


struct MvNormalGamma{T, M <: AbstractArray{T}, L <: AbstractMatrix{T}, A <: Real, B <: Real} <: ContinuousMultivariateDistribution
 
    μ::M # Mean vector
    Λ::L # Precision matrix
    α::A # Shape parameter
    β::B # Rate parameter

    function MvNormalGamma(μ::M, Λ::L, α::A, β::B) where {T, M <: AbstractArray{T}, L <: AbstractMatrix{T}, A <: Real, B <: Real}
        
        # if α <= 0.0; error("Shape parameter must be positive."); end
        # if β <= 0.0; error("Rate parameter must be positive."); end
        
        dimensions = length(μ)
        if size(Λ, 1) != dimensions
            error("Number of rows of precision matrix does not match mean vector length.")
        end
        if size(Λ, 2) != dimensions
            error("Number of columns of precision matrix does not match mean vector length.")
        end

        return new{T,M,L,A,B}(μ,Λ,α,β)
    end
end

BayesBase.dim(d::MvNormalGamma) = length(d.μ)
BayesBase.params(d::MvNormalGamma) = (d.μ, d.Λ, d.α, d.β)
BayesBase.mean(d::MvNormalGamma) = d.μ
BayesBase.precision(d::MvNormalGamma) = d.Λ
BayesBase.cov(d::MvNormalGamma) = cholinv(d.Λ)
BayesBase.shape(d::MvNormalGamma) = d.α
BayesBase.rate(d::MvNormalGamma) = d.β

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
    
    D = BayesBase.dim(left)
    μl, Λl, αl, βl = RxInfer.params(left)
    μr, Λr, αr, βr = RxInfer.params(right)

    Λ = Λl + Λr
    μ = inv(Λl + Λr)*(Λl*μl + Λr*μr)
    α = αl + αr + D/2 - 1
    β = βl + βr + 1/2. *(μl'*Λl*μl + μr'*Λr*μr - (Λl*μl + Λr*μr)'*inv(Λl + Λr)*(Λl*μl + Λr*μr))

    return MvNormalGamma(μ, Λ, α, β)
end