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
default(label="", margin=10Plots.pt)
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
Δt = 0.1

init_state = [0.0, 0.0]
pendulum = SPendulum(init_state = init_state, 
                     mass = sys_mass, 
                     length = sys_length, 
                     damping = sys_damping, 
                     mnoise_sd = sys_mnoise_stdev, 
                     torque_lims = sys_ulims,
                     Δt=Δt)

N = 300
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

    ubuffer = backshift(ubuffer, torques[k])
    buffer[:,k] = [ybuffer; ubuffer]
    ybuffer = backshift(ybuffer, observations[k])

end

p11 = plot(ylabel="angle")
plot!(tsteps, states[1,:], color="blue", label="state")
scatter!(tsteps, observations, color="black", label="measurements")
p12 = plot(xlabel="time [s]", ylabel="torque")
plot!(tsteps, controls[:], color="red")
plot!(tsteps, torques[:], color="purple")
p10 = plot(p11,p12, layout=grid(2,1, heights=[0.7, 0.3]), size=(900,600))

savefig(p10, "experiments/figures/simsys.png")


## Adaptive control

μkmin1 = zeros(M)
Λkmin1 = diageye(M)
αkmin1 = 1.0
βkmin1 = 1.0
mu = 0.0
vu = 1.0
m_star = 0.0
v_star = 1.0

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

k = 10
results = infer(
    model = ARXAgent(μkmin1=μkmin1, 
                     Λkmin1=Λkmin1, 
                     αkmin1=αkmin1, 
                     βkmin1=βkmin1, 
                     mu=mu, 
                     vu=vu,
                     m_star=m_star,
                     v_star=v_star),
    data  = (yk     = observations[k],
             ykmin1 = observations[k-1], 
             ykmin2 = observations[k-2],
             uk     = torques[k],
             ukmin1 = torques[k-1], 
             ukmin2 = torques[k-2]),
)


