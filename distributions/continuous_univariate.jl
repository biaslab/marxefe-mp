import BayesBase
import Distributions: mean, mode, var, std
using ExponentialFamily
using DomainSets
using Optim
using ForwardDiff

function mode(p::ContinuousUnivariateLogPdf{FullSpace{T}}; x0::T=0.0) where T <: Real
    "Mode obtained through maximization."
    
    results = optimize(x -> -p.logpdf(first(x)), [x0], LBFGS(); autodiff=:forward)
    return Optim.minimizer(results)
end

function mode(p::ContinuousUnivariateLogPdf{FullSpace{T}}; limits::Tuple=(0.0,1.0)) where T <: Real
    "Mode obtained through maximization."
    
    p_limited = ContinuousUnivariateLogPdf(ClosedInterval(limits...), p.logpdf)
    return mode(p_limited)
end

function mode(p::ContinuousUnivariateLogPdf{FullSpace{T}}; x0::T=0.0) where T <: Real
    "Mode obtained through maximization."
    
    results = optimize(x -> -p.logpdf(first(x)), [x0], LBFGS(); autodiff=:forward)
    return Optim.minimizer(results)
end

function mode(p::ContinuousUnivariateLogPdf{I}) where I <: Interval
    "Mode obtained through maximization."

    results = optimize(x -> p.logpdf(first(x)), p.domain.left, p.domain.right)
    return Optim.minimizer(results)
end

function mean(p::ContinuousUnivariateLogPdf{I}; num_points::Integer=100) where {I <: Interval}
    "Mean obtained through grid averaging."
    
    points = range(p.domain.left, stop=p.domain.right, length=num_points)
    weights = [exp(p.logpdf(point)) for point in points]
    return sum(points.*weights)/sum(weights)
end

function var(p::ContinuousUnivariateLogPdf{I}; num_points::Integer=100) where {I <: Interval}
    "Variance obtained through grid averaging."

    points = range(p.domain.left, stop=p.domain.right, length=num_points)
    weights = [exp(p.logpdf(point)) for point in points]
    mean = sum(points.*weights)/sum(weights)
    return sum((points .- mean).^2 .*weights)/sum(weights)
end

function std(p::ContinuousUnivariateLogPdf{I}; num_points::Integer=100) where {I <: Interval}
    "Standard deviation obtained through grid averaging."
    return sqrt(var(p, num_points=num_points))
end