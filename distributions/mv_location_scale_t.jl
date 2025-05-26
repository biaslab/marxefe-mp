
import BayesBase
using LinearAlgebra
using Distributions
using RxInfer
using SpecialFunctions



struct MvLocationScaleT{T, N <: Real, M <: AbstractVector{T}, S <: AbstractMatrix{T}} <: ContinuousMultivariateDistribution
 
    η::N # Degrees-of-freedom
    μ::M # Mean vector
    Σ::S # Covariance matrix

    function MvLocationScaleT(η::N, μ::M, Σ::S) where {T, N <: Real, M <: AbstractVector{T}, S <: AbstractMatrix{T}}

        dims = length(μ)
        if η <= dims; error("Degrees of freedom parameter must be larger than the dimensionality."); end
        if dims !== size(Σ,1); error("Dimensionalities of mean and covariance matrix don't match."); end

        return new{T,N,M,S}(η, μ, Σ)
    end
end

BayesBase.params(p::MvLocationScaleT) = (p.η, p.μ, p.Σ)
BayesBase.dim(p::MvLocationScaleT) = length(p.μ)
BayesBase.mean(p::MvLocationScaleT) = p.μ
BayesBase.cov(p::MvLocationScaleT) = p.η > 2 ? p.η/(p.η-2)*p.Σ : error("Degrees of freedom parameter must be larger than 2.")
BayesBase.precision(p::MvLocationScaleT) = inv(cov(p))

function pdf(p::MvLocationScaleT, x)
    d = dims(p)
    η, μ, Σ = params(p)
    return sqrt(1/( (η*π)^d*det(Σ) )) * gamma((η+d)/2)/gamma(η/2) * (1 + 1/η*(x-μ)'*inv(Σ)*(x-μ))^(-(η+d)/2)
end

function logpdf(p::MvLocationScaleT, x)
    d = dims(p)
    η, μ, Σ = params(p)
    return -d/2*log(η*π) - 1/2*logdet(Σ) +loggamma((η+d)/2) -loggamma(η/2) -(η+d)/2*log(1 + 1/η*(x-μ)'*inv(Σ)*(x-μ))
end
