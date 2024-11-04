import BayesBase
using DomainSets

function BayesBase.prod(p1::NormalMeanVariance{T}, p2::ContinuousUnivariateLogPdf{DomainSets.FullSpace{T}}) where T
    
    m = mean(p1)
    v = var( p1)

    logp(u) = -1/2. *log(2π*v) - (u - m)^2/(2v)
    h(u) = logp(u) + p2.logpdf(u)

    return ContinuousUnivariateLogPdf(h)
end