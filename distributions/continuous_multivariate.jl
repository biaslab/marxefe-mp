import BayesBase
import Distributions: mode
using ExponentialFamily
using DomainSets
using Optim
using ForwardDiff

function mode(p::BayesBase.ContinuousMultivariateLogPdf{D}; x0::T=0.0) where {D <: Domain, T <: Real}
    "Mode obtained through maximization."
    
    error("bam")

    results = optimize(x -> -p.logpdf(first(x)), [x0], LBFGS(); autodiff=:forward)
    return Optim.minimizer(results)
end

function BayesBase.prod(::BayesBase.PreserveTypeRightProd, left::MvNormalMeanPrecision, right::ContinuousMultivariateLogPdf)    
    error("boo")
end
