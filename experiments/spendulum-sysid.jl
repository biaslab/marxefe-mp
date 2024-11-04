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
using Plots
default(label="", linewidth=3, margin=10Plots.pt)
includet("../systems/Pendulums.jl"); using .Pendulums

includet("../distributions/mv_normal_gamma.jl")
includet("../distributions/location_scale_tdist.jl")
includet("../distributions/mv_location_scale_tdist.jl")

includet("../nodes/mv_normal_gamma.jl")
includet("../nodes/location_scale_tdist.jl")
includet("../nodes/mv_location_scale_tdist.jl")
includet("../nodes/arxefe.jl")

includet("../rules/mv_normal_gamma/out.jl")
includet("../rules/location_scale_tdist/out.jl")
includet("../rules/mv_location_scale_tdist/out.jl")
includet("../rules/arx_efe/out.jl")
includet("../rules/arx_efe/in.jl")
includet("../rules/arx_efe/parameter.jl")

includet("../src/product.jl")
includet("../src/util.jl")

## System specification

sys_mass = 0.8
sys_length = 0.5
sys_damping = 0.01
sys_mnoise_stdev = 1e-3
sys_ulims = (-10., 10.)
Δt = 0.01

init_state = [0.0, 0.0]
pendulum = SPendulum(init_state = init_state, 
                     mass = sys_mass, 
                     length = sys_length, 
                     damping = sys_damping, 
                     mnoise_sd = sys_mnoise_stdev, 
                     torque_lims = sys_ulims,
                     Δt=Δt)

N = 1_000
tsteps = range(0.0, step=Δt, length=N)                     

A  = 100rand(10)
Ω  = rand(10)*3
controls = mean([A[i]*sin.(Ω[i].*tsteps) for i = 1:10]) ./ 20;

My = 2
Mu = 2
M = My+Mu+1

ybuffer      = zeros(My)
ubuffer      = zeros(Mu+1)
buffer       = zeros(M, N)
states       = zeros(2,N)   
observations = zeros(N)
torques      = zeros(N)

for k in 1:N   

    states[:,k] = pendulum.state
    observations[k] = pendulum.sensor
    step!(pendulum, controls[k])
    torques[k] = pendulum.torque

    global ubuffer = backshift(ubuffer, torques[k])
    xk = [ybuffer; ubuffer]
    buffer[:,k] = xk
    global ybuffer = backshift(ybuffer, observations[k])

end

p11 = plot(ylabel="angle")
plot!(tsteps, states[1,:], color="blue", label="state")
scatter!(tsteps, observations, color="black", label="measurements")
p12 = plot(xlabel="time [s]", ylabel="torque")
plot!(tsteps, controls[:], color="red")
plot!(tsteps, torques[:], color="purple")
p10 = plot(p11,p12, layout=grid(2,1, heights=[0.7, 0.3]), size=(900,600))

savefig(p10, "experiments/figures/simsys.png")


## Offline system identification

μ0 = zeros(M)
Λ0 = diageye(M)
α0 = 1.0
β0 = 1e2

@model function ARXID(y,yprev1,yprev2,u,uprev1,uprev2, μ0,Λ0,α0,β0)

    # Parameter prior
    ζ ~ MvNormalGamma(μ0, Λ0, α0, β0)

    for k in eachindex(y)

        # Autoregressive likelihood
        y[k] ~ ARXEFE(yprev1[k],
                      yprev2[k],
                      u[k],
                      uprev1[k],
                      uprev2[k],
                      ζ)

    end
end

yprev1 = circshift(observations,1); yprev1[1] = 0.0
yprev2 = circshift(yprev1,1); yprev2[1] = 0.0
uprev1 = circshift(torques,1); uprev1[1] = 0.0
uprev2 = circshift(uprev1,1); uprev2[1] = 0.0

results = infer(
    model = ARXID(μ0=μ0, Λ0=Λ0, α0=α0, β0=β0),
    data  = (y = observations,
             yprev1 = yprev1,
             yprev2 = yprev2,
             u = torques,
             uprev1 = uprev1,
             uprev2 = uprev2,),
)

μ_hat,Λ_hat,α_hat,β_hat = BayesBase.params(results.posteriors[:ζ])

ppred_m = zeros(N)
ppred_s = zeros(N)
for k in 1:N
    ppred_m[k] = buffer[:,k]'*μ_hat
    ppred_s[k] = β_hat/α_hat*(buffer[:,k]'*inv(Λ_hat)*buffer[:,k] + 1)
end

p21 = plot(title="offline sysid", xlabel="time", ylabel="angle")
plot!(tsteps, states[1,:], alpha=0.5, color="blue", label="state")
plot!(tsteps, ppred_m, ribbon=ppred_s, color="purple", label="1-step ahead predictions")

savefig(p21, "experiments/figures/offlinesysid_1-step-ahead-predictions.png")

## Online system identification

@model function ARXID(yk,yprev1,yprev2,uk,uprev1,uprev2, μ0,Λ0,α0,β0)

    # Parameter prior
    ζ ~ MvNormalGamma(μ0, Λ0, α0, β0)

    # Autoregressive likelihood
    yk ~ ARXEFE(yprev1,yprev2,uk,uprev1,uprev2,ζ)

end

yprev1 = circshift(observations,1); yprev1[1] = 0.0
yprev2 = circshift(yprev1,1); yprev2[1] = 0.0
uprev1 = circshift(torques,1); uprev1[1] = 0.0
uprev2 = circshift(uprev1,1); uprev2[1] = 0.0

μs = zeros(M,N)
Λs = zeros(M,M,N)
αs = zeros(N)
βs = zeros(N)

for k in 1:N
    results = infer(
        model = ARXID(μ0=μ0, Λ0=Λ0, α0=α0, β0=β0),
        data  = (yk = observations[k],
                 yprev1 = yprev1[k],
                 yprev2 = yprev2[k],
                 uk = torques[k],
                 uprev1 = uprev1[k],
                 uprev2 = uprev2[k],),
    )
    μs[:,k],Λs[:,:,k],αs[k],βs[k] = BayesBase.params(results.posteriors[:ζ])
end

ppred_m = zeros(N)
ppred_s = zeros(N)
for k in 1:N
    ppred_m[k] = buffer[:,k]'*μs[:,k]
    ppred_s[k] = sqrt(βs[k]/αs[k]*(buffer[:,k]'*inv(Λs[:,:,k])*buffer[:,k] + 1))
end

p31 = plot(title="online sysid", xlabel="time", ylabel="angle")
plot!(tsteps, states[1,:], alpha=0.5, color="blue", label="state")
plot!(tsteps, ppred_m, ribbon=ppred_s/100, color="purple", label="1-step ahead predictions")

savefig(p31, "experiments/figures/onlinesysid_1-step-ahead-predictions.png")

