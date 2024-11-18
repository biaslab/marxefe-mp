export Uniform

import BayesBase
using LinearAlgebra
using Distributions


struct Uniform{T} <: ContinuousUnivariateDistribution where {T <: AbstractFloat}
 
    a :: T # Lower limit
    b :: T # Upper limit

    function Uniform(a::F, b::F) where {F <: AbstractFloat}

        if a >= b; error("Lower limit must be smaller than upper limit."); end
        return new{F}(a,b)
    end
end

BayesBase.dim(d::Uniform) = 1
BayesBase.params(d::Uniform) = (d.a, d.b)
BayesBase.mean(d::Uniform) = (d.b-d.a) / 2.
BayesBase.median(d::Uniform) = (d.b-d.a) / 2.
BayesBase.var(d::Uniform) = (d.b-d.a)^2 / 12.
BayesBase.std(d::Uniform) = sqrt(var(d))
BayesBase.precision(d::Uniform) = inv(var(d))
lower(d::Uniform) = d.a
upper(d::Uniform) = d.b

function BayesBase.pdf(dist::Uniform, x::Real)
    if a <= x <= b
        return 1/(dist.b - dist.a)
    else
        return 0.0
    end
end

function BayesBase.logpdf(dist::Uniform, x::Real)
    if a <= x <= b
        return -log(dist.b - dist.a)
    else
        return -Inf
    end
end

BayesBase.default_prod_rule(::Type{<:Uniform}, ::Type{<:Uniform}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::Uniform, right::Uniform)
    
    a = max(left.a,right.a)
    b = min(left.b,right.b)
    
    return Uniform(a,b)
end