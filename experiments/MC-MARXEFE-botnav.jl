# using Pkg
# Pkg.activate("..")
# Pkg.instantiate()

using Revise
using Colors
using Optim
using DomainSets
using JLD2
using ForwardDiff
using ProgressMeter
using LinearAlgebra
using Distributions
using RxInfer; RxInfer.disable_inference_error_hint!()
# using StatsPlots
using Logging; Logging.LogLevel(-1000)
# using Plots; default(label="", grid=false, markersize=3, margin=15Plots.pt)

includet("envs/Robots.jl"); using. Robots
includet("../src/util.jl");
includet("../distributions/matrix_normal_wishart.jl");
includet("../distributions/unboltzmann.jl");
includet("../distributions/mv_location_scale_t.jl");
includet("../distributions/mv_normal_mean_precision.jl")
includet("../nodes/MARX.jl");
includet("../nodes/matrix_normal_wishart.jl");
includet("../rules/MARX/in.jl");
includet("../rules/MARX/out.jl");
includet("../rules/MARX/parameter.jl");
includet("../rules/matrix_normal_wishart/out.jl")



@model function MARX_learning(y_k,y_kmin1,y_kmin2,u_k,u_kmin1,u_kmin2, M_kmin1,Λ_kmin1,Ω_kmin1,ν_kmin1)
    "Update MARX parameters"

    # Prior distribution over MARX parameters
    Φ ~ MatrixNormalWishart(M_kmin1, Λ_kmin1, Ω_kmin1, ν_kmin1)

    # MARX Likelihood
    y_k ~ MARX(y_kmin1,y_kmin2,u_k,u_kmin1,u_kmin2,Φ)

end

@model function MARX_planning(y_tmin1,y_tmin2,u_tmin1,u_tmin2, M_k,Λ_k,Ω_k,ν_k,Υ,m_star,S_star,len_horizon)
    "1-step ahead planning"

    # Posterior distribution over MARX parameters
    Φ   ~ MatrixNormalWishart(M_k,Λ_k,Ω_k,ν_k)

    # Action priors
    u_[1] ~ MvNormalMeanPrecision(zeros(2),Υ)
    u_[2] ~ MvNormalMeanPrecision(zeros(2),Υ)

    # MARX likelihood for t = 1,2
    y_[1] ~ MARX(y_tmin1,y_tmin2,u_[1],u_tmin1,u_tmin2,Φ)
    y_[2] ~ MARX(y_[1],y_tmin1,u_[2],u_[1],u_tmin1,Φ)

    # y_[1] ~ MvNormalMeanCovariance(m_star,S_star)
    # y_[2] ~ MvNormalMeanCovariance(m_star,S_star)

    for t = 3:len_horizon

        u_[t] ~ MvNormalMeanPrecision(zeros(2),Υ)
        y_[t] ~ MARX(y_[t-1],y_[t-2],u_[t],u_[t-1],u_[t-2],Φ)
        # y_[t] ~ MvNormalMeanCovariance(m_star,S_star)
    end
    
    # Goal prior at final horizon point
    y_[len_horizon] ~ MvNormalMeanCovariance(m_star,S_star)
end

posterior_predictive(x_t,M,Λ,Ω,ν,Dx,Dy) = ( ν-Dy+1, M'*x_t, 1/(ν-Dy+1) * Ω * (1 + x_t'*inv(Λ)*x_t) )
function logevidence(y,x,M,Λ,Ω,ν,Dx,Dy)
    η, μ, Σ = posterior_predictive(x,M,Λ,Ω,ν,Dx,Dy)
    return -1/2*(Dy*log(η*π) +logdet(Σ) - 2*logmultigamma(Dy, (η+Dy)/2) + 2*logmultigamma(Dy, (η+Dy-1)/2) + (η+Dy)*log(1 + 1/η*(y-μ)'*inv(Σ)*(y-μ)) )
end

# Trial number (saving id)
experiment_ids = 21:50

# Time
Δt = 0.1
len_trial = 10_000
tsteps = range(0, step=Δt, length=len_trial)
len_horizon = 3;

# Dimensionalities
Mu = 2
My = 2
Dy = 2
Du = Dy
Dx = My*Dy + (Mu+1)*Du
Dz = 4

# Parameters
σ = 1e-6*ones(Dy)
ρ = 1e-3*ones(Dy)

# Limits of controller
global u_lims = (-1.0, 1.0)

# Initial state
z_0 = [0., 0., 0., 0.]

# Goal prior parameters
m_star = [0., 1.]
S_star = 1e-6diagm(ones(Dy))
goal = MvNormalMeanCovariance(m_star, S_star)

# Prior parameters
ν0 = 100
Ω0 = 1e0*diagm(ones(Dy))
Λ0 = 1e-2*diagm(ones(Dx))
M0 = ones(Dx,Dy)/(Dx*Dy)
Υ  = 1e-12*diagm(ones(Dy))

@showprogress for nn in experiment_ids
    @info "Starting experiment $nn"

    # Start robot
    fbot  = FieldBot(ρ,σ, Δt=Δt, control_lims=u_lims)

    # Preallocate
    z_sim = zeros(Dz,len_trial)
    y_sim = zeros(Dy,len_trial)
    u_sim = zeros(Du,len_trial)
    F_sim = zeros(len_trial)
    G_sim = zeros(len_trial)
    D_sim = zeros(len_trial)

    plans_m = zeros(Dy,len_horizon,len_trial)
    plans_S = repeat(diagm(ones(Dy)), outer=[1, 1, len_horizon, len_trial])
    preds_m = zeros(Dy,len_trial+1)
    preds_S = repeat(diagm(ones(Dy)), outer=[1, 1, len_trial+1])

    ybuffer = zeros(Dy,My)
    ubuffer = zeros(Du,Mu+1)

    Ms = zeros(Dx,Dy,len_trial)
    Λs = zeros(Dx,Dx,len_trial)
    Ωs = zeros(Dy,Dy,len_trial)
    νs = zeros(len_trial)

    # Fix starting state
    z_prev  = z_0
    M_kmin1 = M0
    Λ_kmin1 = Λ0
    Ω_kmin1 = Ω0
    ν_kmin1 = ν0

    planres = Vector{Any}(undef,len_trial)

    @showprogress for k in 1:len_trial
    # k = 2

        # global z_prev
        # global M_kmin1
        # global Λ_kmin1
        # global Ω_kmin1
        # global ν_kmin1
        # global ybuffer
        # global ubuffer

        """Interact with env"""

        # Update system with action
        y_sim[:,k], z_sim[:,k] = update(fbot, z_prev, u_sim[:,k])
        z_prev = z_sim[:,k]

        # @info "u = " u_sim[:,k]
        # @info "z = " z_sim[:,k]

        """Infer parameters"""

        # Update MARX parameter belief
        results_learning = infer(
            model = MARX_learning(y_kmin1 = ybuffer[:,1],
                                y_kmin2 = ybuffer[:,2],
                                u_k     = ubuffer[:,1],
                                u_kmin1 = ubuffer[:,2],
                                u_kmin2 = ubuffer[:,3],
                                M_kmin1 = M_kmin1,
                                Λ_kmin1 = Λ_kmin1,
                                Ω_kmin1 = Ω_kmin1,
                                ν_kmin1 = ν_kmin1,),
            data = (y_k = y_sim[:,k],),
        )

        # Track belief
        Ms[:,:,k],Λs[:,:,k],Ωs[:,:,k],νs[k] = params(results_learning.posteriors[:Φ])
        M_kmin1 = Ms[:,:,k]
        Λ_kmin1 = Λs[:,:,k]
        Ω_kmin1 = Ωs[:,:,k]
        ν_kmin1 = νs[k] 

        # Update output buffer
        ybuffer = backshift(ybuffer,y_sim[:,k])

        """Plan and infer actions"""

        inits = @initialization begin
            q(Φ)  = results_learning.posteriors[:Φ]
            q(y_) = vague(MvNormalMeanCovariance,Dy)
            q(u_) = vague(MvNormalMeanCovariance,Du)
        end

        cons = @constraints begin
            q(y_,u_,Φ) = q(y_)q(u_)q(Φ)
            q(y_) = q(y_[begin])..q(y_[end])
            q(u_) = q(u_[begin])..q(u_[end])
            q(u_) :: PointMassFormConstraint()
        end

        # Feed updated beliefs, goal prior params and buffers to planning model
        try
            results_planning = infer(
                model = MARX_planning(M_k         = Ms[:,:,k],
                                        Λ_k         = Λs[:,:,k],
                                        Ω_k         = Ωs[:,:,k],
                                        ν_k         = νs[k],
                                        Υ           = Υ,
                                        m_star      = m_star, 
                                        S_star      = S_star,
                                        len_horizon = len_horizon,),
                data = (y_tmin1 = ybuffer[:,1],
                        y_tmin2 = ybuffer[:,2],
                        u_tmin1 = ubuffer[:,1],
                        u_tmin2 = ubuffer[:,2],),
                initialization = inits,
                constraints = cons,
                iterations = 50, 
                options = (limit_stack_depth=100,),
            )
            planres[k] = results_planning

            # Extract action
            if k < len_trial
                u_sim[:,k+1] = mode(results_planning.posteriors[:u_][end][1])
            end

            # Store output plans
            plans_m[:,:,k] = cat(mode.(results_planning.posteriors[:y_][end])...,dims=2)
            # plans_S[:,:,:,k] = cat(cov.(results_planning.posteriors[:y_][end])...,dims=3)
            
        catch e
            @info "Action selection failed due to" e
        end

        # Update input buffer
        if k < len_trial; ubuffer = backshift(ubuffer,u_sim[:,k+1]); end

        """Predict next observation"""

        x_k = [ybuffer[:]; ubuffer[:]]
        η,μ,Σ = posterior_predictive(x_k,M_kmin1,Λ_kmin1,Ω_kmin1,ν_kmin1,Dx,Dy)
        preds_m[:,k+1] = μ
        preds_S[:,:,k+1] = Σ*η/(η-2)

        # Calculate metrics
        F_sim[k] = -logevidence(y_sim[:,k], x_k,M_kmin1,Λ_kmin1,Ω_kmin1,ν_kmin1,Dx,Dy)
        G_sim[k] = -logpdf(goal,y_sim[:,k])
        D_sim[k] = norm(y_sim[:,k] - m_star,2)
    end

    trialnum = lpad(nn, 3, '0')
    jldsave("experiments/results/MARXEFE-botnav-trialnum$trialnum.jld2"; 
            z_0, z_sim, u_sim, y_sim, F_sim, G_sim, D_sim,
            plans_m, plans_S, preds_m, preds_S, 
            Ms, Λs, Ωs, νs, Υ, goal, 
            Δt, len_trial, len_horizon, Mu, My)
    @info "Saved trial"
end