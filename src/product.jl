import BayesBase
using DomainSets

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
                        left::LocationScaleT, 
                        right::UnivariateGaussianDistributionsFamily{T}) where T <: Real

    logq(u) = logpdf(left,u) + BayesBase.logpdf(right,u)

    return ContinuousUnivariateLogPdf(logq)
end
