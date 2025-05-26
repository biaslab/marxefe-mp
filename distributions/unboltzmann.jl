export unBoltzmann

import BayesBase
import RxInfer
using Optim
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

BayesBase.dim(d::unBoltzmann) = d.D
BayesBase.params(d::unBoltzmann) = d.G

function BayesBase.mode(d::unBoltzmann; u_lims=(-Inf,Inf), time_limit=10., show_trace=false, iterations=1000)
    "Use optimization methods to find maximizer"

    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=show_trace, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=iterations)
    
    results = optimize(d.G, u_lims..., zeros(d.D), Fminbox(LBFGS()), opts, autodiff=:forward)
    return Optim.minimizer(results)
end

function BayesBase.pdf(dist::unBoltzmann, u::Vector)
    "Evaluate exponentiated energy function"
    return exp(-dist.G(u))
end

function BayesBase.logpdf(dist::unBoltzmann, u::Vector)
    "Evaluate energy function"
    return -dist.G(u)
end

BayesBase.default_prod_rule(::Type{<:unBoltzmann}, ::Type{<:unBoltzmann}) = BayesBase.PreserveTypeProd(Distribution)

function BayesBase.prod(::BayesBase.PreserveTypeProd{Distribution}, left::unBoltzmann, right::unBoltzmann)    
    if left.D != right.D; error("Energy functions have different numbers of input variables."); end
    G(u) = left.G(u) + right.G(u)
    return unBoltzmann(G,left.D)
end
