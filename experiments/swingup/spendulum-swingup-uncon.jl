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
using ExponentialFamily
using ExponentialFamilyProjection
using Plots
default(label="", margin=10Plots.pt)
includet("../../systems/Pendulums.jl"); using .Pendulums

includet("../../distributions/mv_normal_gamma.jl")
includet("../../distributions/location_scale_t.jl")
includet("../../distributions/mv_location_scale_t.jl")
includet("../../distributions/continuous_univariate.jl")

includet("../../nodes/mv_normal_gamma.jl")
includet("../../nodes/location_scale_t.jl")
includet("../../nodes/mv_location_scale_t.jl")
includet("../../nodes/arxefe.jl")

includet("../../rules/mv_normal_gamma/out.jl")
includet("../../rules/location_scale_t/out.jl")
includet("../../rules/mv_location_scale_t/out.jl")
includet("../../rules/arx_efe/out.jl")
includet("../../rules/arx_efe/in.jl")
includet("../../rules/arx_efe/parameter.jl")

includet("../../src/product.jl")
includet("../../src/util.jl")

## System specification

sys_mass = 0.8
sys_length = 0.5
sys_damping = 0.01
sys_mnoise_stdev = 1e-3
sys_ulims = (-10., 10.)
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

@model function ARXAgent(yk,ykmin1,ykmin2,uk,ukmin1,ukmin2, μkmin1,Λkmin1,αkmin1,βkmin1,mu,vu,m_star,v_star)

    # Parameter prior
    ζ ~ MvNormalGamma(μkmin1, Λkmin1, αkmin1, βkmin1)

    # Autoregressive likelihood
    yk ~ ARXEFE(ykmin1,ykmin2,uk,ukmin1,ukmin2,ζ)

    # Control prior
    ut ~ NormalMeanVariance(mu,vu)

    # Future likelihood
    yt ~ ARXEFE(yk,ykmin1,ut,uk,ukmin1,ζ)

    # Goal prior
    yt ~ NormalMeanVariance(m_star, v_star)

end

constraints = @constraints begin
    q(ut) :: ProjectedTo(NormalMeanVariance)
    q(yt) :: ProjectedTo(NormalMeanVariance)
end


len_trial = 100
My = 2
Mu = 2
M = My+Mu+1
μ_kmin1 = zeros(M)
Λ_kmin1 = diageye(M)
α_kmin1 = 1.0
β_kmin1 = 1.0
m_u = 0.0
v_u = 1.0
m_star = 0.0
v_star = 0.5

states       = zeros(2, len_trial)
observations = zeros(len_trial)
torques      = zeros(len_trial)
μs           = zeros(M,len_trial)
Λs           = zeros(M,M,len_trial)
αs           = zeros(len_trial)
βs           = zeros(len_trial)
py           = []
pu           = []

for k in M:len_trial

    states[:,k] = pendulum.state
    observations[k] = pendulum.sensor

    results = infer(
        model        = ARXAgent(μkmin1=μ_kmin1,
                                Λkmin1=Λ_kmin1,
                                αkmin1=α_kmin1,
                                βkmin1=β_kmin1,
                                mu=m_u,
                                vu=v_u,
                                m_star=m_star,
                                v_star=v_star),
        data         = (yk     = observations[k],
                        ykmin1 = observations[k-1], 
                        ykmin2 = observations[k-2],
                        uk     = torques[k],
                        ukmin1 = torques[k-1], 
                        ukmin2 = torques[k-2]),
        # constraints  = constraints,
        # iterations   = 10,
        # showprogress = true,
        # returnvars   = (yt = KeepLast(), 
                        # ut = KeepLast(),
                        # ζ  = KeepLast(),),
    )

    # Take action
    push!(pu, results.posteriors[:ut])
    action = mode(results.posteriors[:ut])
    step!(pendulum, action)
    torques[k] = pendulum.torque

    # Track predictions
    push!(py, results.posteriors[:yt])

    # Track parameters
    μs[:,k]   = μ_kmin1 = mean(results.posteriors[:ζ])
    Λs[:,:,k] = Λ_kmin1 = precision(results.posteriors[:ζ])
    αs[k]     = α_kmin1 = shape(results.posteriors[:ζ])
    βs[k]     = β_kmin1 = rate(results.posteriors[:ζ])

end

tsteps = range(0, step=Δt, length=len_trial)

p101 = plot(xlabel="time", ylabel="angle")
scatter!(tsteps, observations, label="observations")
plot!(collect(tsteps[M:end]), mean.(py), ribbon=std.(py), label="predictions")

p102 = plot(xlabel="time", ylabel="control", ylims=sys_ulims)
plot!(tsteps, torques, label="torques")

plot(p101, p102, layout=(2,1), size=(500,1000))
savefig("experiments/figures/swingup-trial.png")


