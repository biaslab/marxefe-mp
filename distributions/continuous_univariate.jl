import BayesBase
import Distributions: mean, mode
using ExponentialFamily
using DomainSets
using Optim
using ForwardDiff

function mode(p::ContinuousUnivariateLogPdf{FullSpace{T}}; x0::T=0.0) where T <: Real
    "Mode obtained through maximization."
    
    results = optimize(x -> -p.logpdf(first(x)), [x0], LBFGS(); autodiff=:forward)
    return Optim.minimum(results)
end

function mode(p::ContinuousUnivariateLogPdf{FullSpace{T}}; x0::T=0.0) where T <: Real
    "Mode obtained through maximization."
    
    results = optimize(x -> -p.logpdf(first(x)), [x0], LBFGS(); autodiff=:forward)
    return Optim.minimum(results)
end

# function mode(p::ContinuousUnivariateLogPdf{HalfLine}; x0::Real=0.0)
#     "Mode obtained through maximization."
    
#     results = optimize(x -> -p.logpdf(first(x)), 0.0, Inf, [x0], LBFGS(); autodiff=:forward)
#     return Optim.minimum(results)
# end