export unBoltzmann

import BayesBase
import RxInfer
using Optim
using Logging
using LinearAlgebra
using Distributions
using SpecialFunctions


struct unBoltzmann <: ContinuousMultivariateDistribution
 
    G::Function # Energy function
    D::Integer  # Number of input arguments of energy function

    function unBoltzmann(G::Function, D::Integer)
        return new(G,D)
    end
end

BayesBase.ndims(d::unBoltzmann) = d.D

function BayesBase.mode(d::unBoltzmann; u_lims=(-Inf,Inf), time_limit=0.5, show_trace=false, iterations=3)
    "Use optimization methods to find maximizer"

    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=show_trace, 
                         allow_f_increases=true, 
                         outer_iterations=iterations,
                         iterations=1)

    @debug opts
    gradG(J,u) = ForwardDiff.gradient!(J,d.G,u)
    results = optimize(d.G, gradG, repeat([u_lims[1]],d.D), repeat([u_lims[2]],d.D), 1e-8*randn(d.D), Fminbox(LBFGS()), opts)
    # results = optimize(d.G, 1e-8*randn(d.D), LBFGS(), opts, autodiff=:forward)
    return clamp!(Optim.minimizer(results), u_lims...)
end

function pdf(dist::unBoltzmann, u::Vector)
    "Evaluate exponentiated energy function"
    return exp(-dist.G(u))
end

function logpdf(dist::unBoltzmann, u::Vector)
    "Evaluate energy function"
    return -dist.G(u)
end

BayesBase.default_prod_rule(::Type{<:unBoltzmann}, ::Type{<:unBoltzmann})      = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:AbstractMvNormal}, ::Type{<:unBoltzmann}) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:unBoltzmann}, ::Type{<:AbstractMvNormal}) = BayesBase.ClosedProd()

function BayesBase.prod(::BayesBase.ClosedProd, left::unBoltzmann, right::unBoltzmann)    
    if left.D != right.D; error("Dimensionalities of energy functions do not match."); end
    G(u) = left.G(u) + right.G(u)
    return unBoltzmann(G,right.D)
end

function BayesBase.prod(::BayesBase.ClosedProd, left::AbstractMvNormal, right::unBoltzmann)    
    if ndims(left) != right.D; error("Dimensionality of Gaussian and number of inputs of energy function do not match."); end
    G(u) = BayesBase.logpdf(left,u) + right.G(u)
    return unBoltzmann(G,right.D)
end

function BayesBase.prod(::BayesBase.ClosedProd, left::unBoltzmann, right::AbstractMvNormal)    
    if left.D != ndims(right); error("Dimensionality of Gaussian and number of inputs of energy function do not match."); end
    G(u) = left.G(u) + BayesBase.logpdf(right,u)
    return unBoltzmann(G,left.D)
end


# BayesBase.default_prod_rule(::Type{<:unBoltzmann}, ::Type{<:unBoltzmann}) = BayesBase.PreserveTypeProd(Distribution)

# function BayesBase.prod(::BayesBase.PreserveTypeProd{Distribution}, left::unBoltzmann, right::unBoltzmann)    
#     if left.D != right.D; error("Energy functions have different numbers of input variables."); end
#     G(u) = left.G(u) + right.G(u)
#     return unBoltzmann(G,left.D)
# end

# function BayesBase.prod(::BayesBase.PreserveTypeRightProd, left::MvNormalMeanPrecision, right::unBoltzmann)    
#     if ndims(left) != right.D; error("Dimensionality of Gaussian and number of inputs of energy function do not match."); end
#     G(u) = BayesBase.logpdf(left,u) + right.G(u)
#     return unBoltzmann(G,right.D)
# end

# function BayesBase.prod(::BayesBase.PreserveTypeLeftProd, left::unBoltzmann, right::MvNormalMeanPrecision)    
#     if left.D != ndims(right); error("Dimensionality of Gaussian and number of inputs of energy function do not match."); end
#     G(u) = left.G(u) + BayesBase.logpdf(right,u)
#     return unBoltzmann(G,left.D)
# end