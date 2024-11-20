import BayesBase: prod
using DomainSets
using IntervalSets

function default_prod_rule(::ContinuousUnivariateLogPdf{FullSpace{T}}, 
                           ::UnivariateGaussianDistributionsFamily{T}) where {T} <: Real
    return PreserveTypeProd(T)
end

function BayesBase.prod(::PreserveTypeLeftProd, 
                        left::ContinuousUnivariateLogPdf{FullSpace{T}}, 
                        right::UnivariateGaussianDistributionsFamily{T}) where T <: Real

    logq(u) = left.logpdf(u) + BayesBase.logpdf(right,u)

    return ContinuousUnivariateLogPdf(logq)
end

function BayesBase.prod(::PreserveTypeRightProd, 
                        left::UnivariateGaussianDistributionsFamily{T},
                        right::ContinuousUnivariateLogPdf{FullSpace{T}}) where T <: Real

    logq(u) = BayesBase.logpdf(left,u) + right.logpdf(u)

    return ContinuousUnivariateLogPdf(logq)
end

function BayesBase.prod(::GenericProd, 
                        left::ContinuousUnivariateLogPdf{FullSpace{T}}, 
                        right::UnivariateGaussianDistributionsFamily{T}) where T <: Real

    logq(u) = left.logpdf(u) + BayesBase.logpdf(right,u)

    return ContinuousUnivariateLogPdf(logq)
end

function BayesBase.prod(::GenericProd, 
                        left::UnivariateGaussianDistributionsFamily{T},
                        right::ContinuousUnivariateLogPdf{FullSpace{T}}) where T <: Real

    logq(u) = BayesBase.logpdf(left,u) + right.logpdf(u)

    return ContinuousUnivariateLogPdf(logq)
end

function BayesBase.prod(::GenericProd, 
                        left::LocationScaleT, 
                        right::UnivariateGaussianDistributionsFamily{T}) where T <: Real

    # Gaussian approximation to LocationScaleT  
    ν,μ,σ = params(left)                        
    left_normal = NormalMeanVariance(μ, σ^2*(ν/(ν-2)))

    return BayesBase.prod(ClosedProd(), left_normal, right)
end

function BayesBase.prod(::GenericProd, 
                        left::Uniform,
                        right::ContinuousUnivariateLogPdf{FullSpace{T}}) where T <: Real

    logq(x) = BayesBase.logpdf(left,x) + right.logpdf(x)

    return ContinuousUnivariateLogPdf(Interval(left.a..left.b), logq)
end