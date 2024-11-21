"""
Experiment ARX-EFE MP

Wouter M. Kouw
"""

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using LinearAlgebra
using Distributions
using RxInfer
# using ExponentialFamily
# using ExponentialFamilyProjection
using Plots
default(label="", margin=10Plots.pt)
includet("../../systems/Pendulums.jl"); using .Pendulums

includet("../../distributions/mv_normal_gamma.jl")
includet("../../distributions/location_scale_t.jl")
includet("../../distributions/mv_location_scale_t.jl")
includet("../../distributions/continuous_univariate.jl")
includet("../../distributions/uniform.jl")

includet("../../nodes/mv_normal_gamma.jl")
includet("../../nodes/location_scale_t.jl")
includet("../../nodes/mv_location_scale_t.jl")
includet("../../nodes/arxefe.jl")
includet("../../nodes/nuv_box.jl")
includet("../../nodes/uniform.jl")

includet("../../rules/mv_normal_gamma/out.jl")
includet("../../rules/location_scale_t/out.jl")
includet("../../rules/mv_location_scale_t/out.jl")
includet("../../rules/arx_efe/out.jl")
includet("../../rules/arx_efe/in.jl")
includet("../../rules/arx_efe/parameter.jl")
includet("../../rules/NUV_box/out.jl")
includet("../../rules/NUV_box/sigma.jl")
includet("../../rules/uniform/out.jl")

includet("../../src/product.jl")
includet("../../src/util.jl")

## System specification

sys_mass = 0.8
sys_length = 0.5
sys_damping = 0.01
sys_mnoise_stdev = 1e-1
sys_ulims = (-100., 100.)
Δt = 0.05

init_state = [0.0, 0.0]
pendulum = SPendulum(init_state = init_state, 
                     mass = sys_mass, 
                     length = sys_length, 
                     damping = sys_damping, 
                     mnoise_sd = sys_mnoise_stdev, 
                     torque_lims = sys_ulims,
                     Δt=Δt)


## Adaptive control

@model function ARXAgent(yk,ykmin1,ykmin2,uk,ukmin1,ukmin2, μkmin1,Λkmin1,αkmin1,βkmin1,m_star,v_star)

    # Parameter prior
    ζ ~ MvNormalGamma(μkmin1, Λkmin1, αkmin1, βkmin1)

    # Autoregressive likelihood
    yk ~ ARXEFE(ykmin1,ykmin2,uk,ukmin1,ukmin2,ζ)

    # Control prior
    ut ~ Uniform(sys_ulims[1], sys_ulims[2])

    # Future likelihood
    yt ~ ARXEFE(yk,ykmin1,ut,uk,ukmin1,ζ)

    # Goal prior
    yt ~ NormalMeanVariance(m_star, v_star)

end

len_trial = 300
My = 2
Mu = 2
M = My+Mu+1
μ_kmin1 = zeros(M)
Λ_kmin1 = 1e-1diageye(M)
α_kmin1 = 10.0
β_kmin1 = 1e-2
m_star  = 0.5
v_star  = 0.5

states       = zeros(2, len_trial)
observations = zeros(len_trial)
torques      = zeros(len_trial+1)
μs           = zeros(M,len_trial)
Λs           = zeros(M,M,len_trial)
αs           = zeros(len_trial)
βs           = zeros(len_trial)
my           = []
py           = []
pu           = []
results_     = []

observations[1:M] = 1e-6*randn(M)

for k in M:len_trial

    # Output predictions
    xk = [observations[k-1:-1:k-2]; torques[k:-1:k-2]]
    ν = 2α_kmin1
    μ = μ_kmin1'*xk
    σ = sqrt(β_kmin1/α_kmin1*(xk'*inv(Λ_kmin1)*xk + 1))
    myt = LocationScaleT(ν,μ,σ)
    push!(my, myt)

    # Track system
    states[:,k] = pendulum.state
    observations[k] = pendulum.sensor

    # Infer parameters,action
    results = infer(
        model          = ARXAgent(μkmin1=μ_kmin1,
                                  Λkmin1=Λ_kmin1,
                                  αkmin1=α_kmin1,
                                  βkmin1=β_kmin1,
                                  m_star=m_star,
                                  v_star=v_star),
        data           = (yk     = observations[k],
                          ykmin1 = observations[k-1], 
                          ykmin2 = observations[k-2],
                          uk     = torques[k],
                          ukmin1 = torques[k-1], 
                          ukmin2 = torques[k-2]),
        options        = (limit_stack_depth = 100,),
    )
    push!(results_, results)
    push!(py, results.posteriors[:yt])
    push!(pu, results.posteriors[:ut])

    # Take action
    put = results.posteriors[:ut]
    optres = optimize(x -> put.logpdf(first(x)), put.domain.left, put.domain.right)
    action = Optim.minimizer(optres)
    step!(pendulum, action)
    torques[k+1] = pendulum.torque
    
    # Track variables
    μs[:,k]   = μ_kmin1 = mean(results.posteriors[:ζ])
    Λs[:,:,k] = Λ_kmin1 = precision(results.posteriors[:ζ])
    αs[k]     = α_kmin1 = shape(results.posteriors[:ζ])
    βs[k]     = β_kmin1 = rate(results.posteriors[:ζ])

end

tsteps = range(0, step=Δt, length=len_trial)

p101 = plot(xlabel="time", ylabel="angle", ylims=(-1.,2.))
plot!(tsteps, repeat([m_star], len_trial), ribbon=repeat([v_star], len_trial), color=:green, fillalpha=0.5, label="goal_prior")
scatter!(tsteps, observations, color=:black, label="observations")
plot!(collect(tsteps[M:end]), mean.(my), ribbon=std.(my), color=:orange, label="yₜ messages")
plot!(collect(tsteps[M:end]), mean.(py), ribbon=std.(py), color=:purple, label="yₜ marginals")
savefig("experiments/swingup/figures/swingup-outputpredictions.png")

p102 = plot(xlabel="time", ylabel="control", ylims=[sys_ulims[1], sys_ulims[2]])
urange = range(sys_ulims[1], stop=sys_ulims[2], length=100)
PU = hcat([put.logpdf.(urange) for put in pu]...)
heatmap!(collect(tsteps)[M:end], urange, PU, colormap=:jet, label="Control posteriors")
scatter!(collect(tsteps)[M:end], torques[M:len_trial], color=:black, linewidth=4, label="torques")
savefig("experiments/swingup/figures/swingup-controls.png")

# plot(p101, p102, layout=(2,1), size=(500,1000))
# savefig("experiments/swingup/figures/swingup-trial.png")


function G(u; k=1)
    x = [observations[k-1:-1:k-2]; u; torques[k-1:-1:k-2]]
    return 1/(2v_star)*(βs[k]/(αs[k]-1)*(1+x'*inv(Λs[:,:,k])*x) + (μs[:,k]'*x - m_star)^2 ) - 1/2*log(1+x'*inv(Λs[:,:,k])*x)
end

function CE(u; k=1)
    x = [observations[k-1:-1:k-2]; u; torques[k-1:-1:k-2]]
    return 1/(2v_star)*(βs[k]/(αs[k]-1)*(1+x'*inv(Λs[:,:,k])*x) + (μs[:,k]'*x - m_star)^2 )
end

function MI(u; k=1)
    x = [observations[k-1:-1:k-2]; u; torques[k-1:-1:k-2]]
    return - 1/2*log(1+x'*inv(Λs[:,:,k])*x)
end

xx = range(sys_ulims[1], stop=sys_ulims[2], length=100)
urange = range(sys_ulims[1], stop=sys_ulims[2], length=100)
CE_ = hcat([CE.(urange, k=k) for k in M:len_trial]...)
heatmap(collect(tsteps)[M:len_trial], urange, CE_, colormap=:jet, label="Control posteriors")