""" 
Adaptive model-predictive control for mobile robot navigation

Wouter M. Kouw, February 2025
"""

using Pkg
Pkg.activate("..")
Pkg.instantiate()

using Revise
using Colors
using Optim
using JLD2
using MeshGrid
using ForwardDiff
using ProgressMeter
using LinearAlgebra
using Distributions
using StatsPlots
using Plots
default(label="", grid=false, markersize=3, margin=15Plots.pt)

includet("envs/Robots.jl"); using. Robots
includet("baselines/MARXControllers.jl"); using. MARXControllers
includet("../src/util.jl");


# Trial number (saving id)
experiment_ids = 1

# Time
Δt = 0.2
len_trial = 1000
tsteps = range(0, step=Δt, length=len_trial)
len_horizon = 4;

# Dimensionalities
Mu = 2
My = 2
Dy = 2
Du = Dy
Dx = My*Dy + Mu*Du
Dz = 4

# Setpoint (desired observation)
m_star = [1., 1.]
S_star = 0.5*diagm(ones(Dy))
goal = MvNormal(m_star, S_star)

# Parameters
σ = 1e-4*ones(Dy)
ρ = 1e-3*ones(Dy)

# Prior parameters
ν0 = 100
Ω0 = 1e0*diagm(ones(Dy))
Λ0 = 1e-2*diagm(ones(Dx))
M0 = ones(Dx,Dy)/(Dx*Dy)
Υ  = 1e-12*diagm(ones(Dy));

# Limits of controller
u_lims = (-1.0, 1.0)
opts = Optim.Options(time_limit=1.,
                     iterations=100)

# Initial state
z_0 = [-1., -1., 0., 0.]

# @showprogress for nn in experiment_ids
    nn = experiment_ids

    # Start robot
    fbot  = FieldBot(ρ,σ, Δt=Δt, control_lims=u_lims)

    # Start agent
    agent = MARXController(M0,Λ0,Ω0,ν0,Υ, goal, Dy=Dy, Du=Du, delay_inp=Mu, delay_out=My, time_horizon=len_horizon)

    # Preallocate
    z_sim  = zeros(Dz,len_trial)
    y_sim  = zeros(Dy,len_trial)
    u_sim  = zeros(Du,len_trial)
    F_sim  = zeros(len_trial)
    G_sim  = zeros(len_trial)
    preds_m = zeros(Dy,len_trial)
    preds_S = repeat(diagm(ones(Dy)), outer=[1, 1, len_trial])

    Ms = zeros(Dx,Dy,len_trial)
    Λs = zeros(Dx,Dx,len_trial)
    Ωs = zeros(Dy,Dy,len_trial)
    νs = zeros(len_trial)

    # Initial state
    z_sim[:,1] = z_0

    @showprogress for k in 2:len_trial
        
        "Interact with environment"

        # Update system with selected control
        y_sim[:,k], z_sim[:,k] = update(fbot, z_sim[:,k-1], u_sim[:,k-1])
                
        "Parameter estimation"

        # Update parameters
        MARXControllers.update!(agent, y_sim[:,k], u_sim[:,k-1])

        # Store
        Ms[:,:,k] = agent.M
        Λs[:,:,k] = agent.Λ
        Ωs[:,:,k] = agent.Ω
        νs[k]     = agent.ν

        # Track free energy
        F_sim[k] = agent.free_energy

        # Track goal alignment
        G_sim[k] = -logpdf(goal, y_sim[:,k])
        
        "Planning"
        
        # Call minimizer using constrained L-BFGS procedure
        G(u::AbstractVector) = MPC(agent, u)
        results = Optim.optimize(G, u_lims[1], u_lims[2], zeros(Du*len_horizon), Fminbox(LBFGS()), opts; autodiff=:forward)
        
        # Extract minimizing control
        policy = Optim.minimizer(results)
        u_sim[:,k] = policy[1:Du]
        
    end

    trialnum = lpad(nn, 3, '0')
    jldsave("experiments/results/MARXMPC-botnav-trialnum$trialnum.jld2"; F_sim, G_sim, z_0, z_sim, u_sim, y_sim, Ms, Λs, Ωs, νs, Υ, goal, Δt, len_trial, len_horizon, Mu, My)

# end