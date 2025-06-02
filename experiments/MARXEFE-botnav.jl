using Pkg
Pkg.activate("..")
Pkg.instantiate()

using Revise
using Colors
using Optim
using DomainSets
using JLD2
using ForwardDiff
using ProgressMeter
using LinearAlgebra
using Distributions
using StatsPlots
using Logging; Logging.LogLevel(-1000)
using Plots; default(label="", grid=false, markersize=3, margin=15Plots.pt)
using RxInfer; #RxInfer.disable_inference_error_hint!()

includet("envs/Robots.jl"); using. Robots
includet("../src/util.jl");
includet("../distributions/matrix_normal_wishart.jl");
includet("../distributions/mv_location_scale_t.jl");
includet("../distributions/unboltzmann.jl");
includet("../nodes/MARX.jl");
includet("../nodes/matrix_normal_wishart.jl");
includet("../rules/MARX/in.jl");
includet("../rules/MARX/out.jl");
includet("../rules/MARX/parameter.jl");
includet("../rules/matrix_normal_wishart/out.jl")

# Trial number (saving id)
trialnum = 01

# Time
Δt = 0.1
len_trial = 100
tsteps = range(0, step=Δt, length=len_trial)
len_horizon = 4;

# Dimensionalities
Mu = 2
My = 2
Dy = 2
Du = Dy
Dx = My*Dy + (Mu+1)*Du
Dz = 4

# Parameters
σ = 1e-4*ones(Dy)
ρ = 1e-3*ones(Dy)

# Limits of controller
global u_lims = (-3.0, 3.0)

# Initial state
z_0 = [-1., -1., 0., 0.]

# Goal prior parameters
m_star = [1., 1.]
S_star = 1e-3diagm(ones(Dy))
goal = MvNormal(m_star, S_star)

# Prior parameters
ν0 = 100
Ω0 = 1e0*diagm(ones(Dy))
Λ0 = 1e-2*diagm(ones(Dx))
M0 = ones(Dx,Dy) / (Dy*Dx)
Υ  = 1e-12*diagm(ones(Dy));
# M0,Λ0,Ω0,ν0 = params(results.posteriors[:Φ])

# Start robot
fbot  = FieldBot(ρ,σ, Δt=Δt, control_lims=u_lims)


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
    u_[1] ~ MvNormalMeanPrecision(zeros(Du),Υ)
    u_[2] ~ MvNormalMeanPrecision(zeros(Du),Υ)

    # MARX likelihood for t = 1,2
    y_[1] ~ MARX(y_tmin1,y_tmin2,u_[1],u_tmin1,u_tmin2,Φ)
    y_[2] ~ MARX(y_[1],y_tmin1,u_[2],u_[1],u_tmin1,Φ)

    for t = 3:len_horizon

        u_[t] ~ MvNormalMeanPrecision(zeros(Du),Υ)
        y_[t] ~ MARX(y_[t-1],y_[t-2],u_[t],u_[t-1],u_[t-2],Φ)

    end

    # Goal prior at final horizon point
    y_[len_horizon] ~ MvNormalMeanCovariance(m_star,S_star)
    
end

posterior_predictive(x_t,M,Λ,Ω,ν,Dx,Dy) = (ν-Dy+1, M'*x_t, 1/(ν-Dy+1)*Ω*(1+x_t'*inv(Λ)*x_t))

# Preallocate
z_sim   = zeros(Dz,len_trial)
y_sim   = zeros(Dy,len_trial)
u_sim   = zeros(Du,len_trial)
# F_sim   = zeros(len_trial)

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
z_sim[:,1] = z_0
u_sim[:,1] = clamp!(randn(2), u_lims...)
y_sim[:,1] = emit(fbot, z_sim[:,1])
ybuffer    = backshift(ybuffer,y_sim[:,1])
ubuffer    = backshift(ubuffer,u_sim[:,1])
Ms[:,:,1]  = M0
Λs[:,:,1]  = Λ0
Ωs[:,:,1]  = Ω0
νs[1]      = ν0

results_learning = []
results_planning = []

@info "Starting trial."
for k in 2:len_trial
    @info "step = $k / $len_trial"

    """Interact with env"""

    # Update system with action
    y_sim[:,k], z_sim[:,k] = update(fbot, z_sim[:,k-1], u_sim[:,k-1])

    """Infer parameters"""

    # Update input buffer
    ubuffer = backshift(ubuffer,u_sim[:,k-1])

    # Update MARX parameter belief
    results_learning = infer(
        model = MARX_learning(y_kmin1 = ybuffer[:,1],
                              y_kmin2 = ybuffer[:,2],
                              u_k     = ubuffer[:,1],
                              u_kmin1 = ubuffer[:,2],
                              u_kmin2 = ubuffer[:,3],
                              M_kmin1 = Ms[:,:,k-1],
                              Λ_kmin1 = Λs[:,:,k-1],
                              Ω_kmin1 = Ωs[:,:,k-1],
                              ν_kmin1 = νs[k-1]),
        data = (y_k = y_sim[:,k],),
    )

    # Track belief
    Ms[:,:,k],Λs[:,:,k],Ωs[:,:,k],νs[k] = params(results_learning.posteriors[:Φ])

    # Update output buffer
    ybuffer = backshift(ybuffer,y_sim[:,k])

    """Plan and infer actions"""

    inits = @initialization begin
        q(Φ)  = results_learning.posteriors[:Φ]
        μ(Φ)  = results_learning.posteriors[:Φ]
        q(y_) = repeat([vague(MvNormalMeanCovariance,Dy)], len_horizon)
        μ(y_) = repeat([vague(MvNormalMeanCovariance,Dy)], len_horizon)
        q(u_) = repeat([vague(MvNormalMeanCovariance,Du)], len_horizon)
        μ(u_) = repeat([vague(MvNormalMeanCovariance,Du)], len_horizon)
    end

    # constraints = @constraints begin
    #     q(y_,u_,Φ) = q(y_)q(u_)q(Φ)
    # end

    # Feed updated beliefs, goal prior params and buffers to planning model
    try
        @time results_planning = infer(
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
            constraints = MeanField(),
            iterations = 2, 
            showprogress = true,
            options = (limit_stack_depth=100,),
        )

        # Extract action
        u_sim[:,k] = mode(results_planning.posteriors[:u_][end][1], u_lims=u_lims)

        # Store output plans
        plans_m[:,:,k] = cat(mean.(results_planning.posteriors[:y_][end])...,dims=2)
        plans_S[:,:,:,k] = cat(cov.(results_planning.posteriors[:y_][end])...,dims=3)
    
    catch
        @info "fail"
        u_sim[:,k] = zeros(Dy)
    end
    @debug u_sim[:,k]

    """Make predictions for future"""

    x_k = [ybuffer[:]; ubuffer[:]]
    η,μ,Σ = posterior_predictive(x_k,Ms[:,:,k],Λs[:,:,k],Ωs[:,:,k],νs[k],Dx,Dy)
    preds_m[:,k+1] = μ
    preds_S[:,:,k+1] = Σ*η/(η-2)

end

# Save 
jldsave("experiments/results/MARXEFE-botnav-trialnum$trialnum.jld2"; z_sim, u_sim, y_sim, Ms, Λs, Ωs, νs, Υ, plans_m, plans_S, preds_m, preds_S)

# Check actions
plot(u_sim')

# Plot trajectories
twin = 3:len_trial
scatter([z_0[1]], [z_0[2]], label="start", color="green", markersize=5)
scatter!([mean(goal)[1]], [mean(goal)[2]], label="goal", color="red", markersize=5)
covellipse!(mean(goal), cov(goal), n_std=1., linewidth=3, fillalpha=0.01, linecolor="red", color="red")
scatter!(y_sim[1,twin], y_sim[2,twin], alpha=0.5, label="observations", color="black")
plot!(z_sim[1,twin], z_sim[2,twin], label="system path", color="blue")
for kk = twin
    covellipse!(preds_m[:,kk], preds_S[:,:,kk], n_std=1, alpha=0.001, fillalpha=0.0001, color="purple")
end
plot!(preds_m[1,twin], preds_m[2,twin], label="predictions", color="purple")
# plot!(xlims=(-3,3), ylims=(-3,3))

# Plot plans at a certain timepoint
tpoint = 80
scatter([z_0[1]], [z_0[2]], label="start", color="green", markersize=5)
scatter!([mean(goal)[1]], [mean(goal)[2]], label="goal", color="red", markersize=5)
# covellipse!(mean(goal), cov(goal), n_std=1., linewidth=1, fillalpha=0.1, linecolor="red", color="red")
# scatter!(y_sim[1,tpoint], y_sim[2,tpoint], alpha=0.5, label="observations", color="black")
plot!(z_sim[1,1:tpoint], z_sim[2,1:tpoint], linewidth=3, label="system path", color="blue")
for tt in 1:len_horizon
    scatter!([plans_m[1,tt,tpoint]], [plans_m[2,tt,tpoint]], label="y_$tt", color="purple")
    covellipse!(plans_m[:,tt,tpoint], plans_S[:,:,tt,tpoint], n_std=1, alpha=0.1, fillalpha=0.5^tt, color="purple")
end
plot!()