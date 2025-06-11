export unBoltzmann

import BayesBase
import RxInfer
using Optim
using DomainSets
using LinearAlgebra
using Distributions
using SpecialFunctions


struct unBoltzmann <: ContinuousMultivariateDistribution
 
    G::Function     # Energy function
    N::Integer      # Number of input arguments of energy function
    D::Rectangle    # Support of Boltzmann distribution

    function unBoltzmann(G::Function, N::Integer, D::Rectangle)
        return new(G,N,D)
    end
end

BayesBase.ndims(d::unBoltzmann) = d.N
BayesBase.support(d::unBoltzmann) = d.D

function BayesBase.mode(dist::unBoltzmann; time_limit=0.5, show_trace=false, iterations=3)
    "Use optimization methods to find maximizer"
    
    if ndims(dist) == 2
        @debug "Starting quantized mode calculation"

        u1_lims = (support(dist).a[1], support(dist).b[1])
        u2_lims = (support(dist).a[2], support(dist).b[2])

        num_u1 = 51
        num_u2 = 51
        u1 = range(u1_lims[1], stop=u1_lims[2], length=num_u1)
        u2 = range(u2_lims[1], stop=u2_lims[2], length=num_u2)
        field = zeros(num_u1,num_u2)
        for (ii,ui) in enumerate(u1)
            for (jj,uj) in enumerate(u2)
                field[ii,jj] = dist.G([ui,uj])
            end
        end
        u_star = argmin(field)
        return [u1[u_star[1]], u2[u_star[2]]]
    else
        error("Not implemented yet")

        # opts = Optim.Options(time_limit=time_limit, 
        #                      show_trace=show_trace, 
        #                      allow_f_increases=true, 
        #                      outer_iterations=iterations,
        #                      iterations=1)

        # @debug opts
        # gradG(J,u) = ForwardDiff.gradient!(J,d.G,u)
        # results = optimize(d.G, gradG, repeat([u_lims[1]],d.D), repeat([u_lims[2]],d.D), 1e-8*randn(d.D), Fminbox(LBFGS()), opts)
        # # results = optimize(d.G, 1e-8*randn(d.D), LBFGS(), opts, autodiff=:forward)
        # return clamp!(Optim.minimizer(results), u_lims...)
    end
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
    if left.N != right.N; error("Dimensionalities of energy functions do not match."); end
    G(u) = left.G(u) + right.G(u)
    return unBoltzmann(G,right.N, intersectdomain(left.D, right.D))
end

function BayesBase.prod(::BayesBase.ClosedProd, left::AbstractMvNormal, right::unBoltzmann)    
    if ndims(left) != right.N; error("Dimensionality of Gaussian and number of inputs of energy function do not match."); end
    G(u) = -BayesBase.logpdf(left,u) + right.G(u)
    return unBoltzmann(G,right.N,right.D)
end

BayesBase.prod(::BayesBase.ClosedProd, left::unBoltzmann, right::AbstractMvNormal) = BayesBase.prod(ClosedProd(), right, left)    


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