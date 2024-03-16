"""
Experiment description


Wouter M. Kouw
2024-mar-11
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
default(label="")
includet("../systems/Pendulums.jl"); using .Pendulums
# includet("../src/nuv_box.jl");
# includet("../src/diode.jl");
# includet("../src/buffer.jl");
includet("../nodes/mv_normal_gamma.jl")
includet("../nodes/arx.jl")

## System specification

sys_mass = 1.0
sys_length = 1.0
sys_damping = 0.00
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

N = 100
tsteps = range(0.0, step=Δt, length=N)                     

A  = rand(10)*300 .- 100
Ω  = rand(10)*3
controls = mean([A[i]*sin.(Ω[i].*tsteps) for i = 1:10]) ./ 20;

states = zeros(2,N)
observations = zeros(N)
torques = zeros(N)

for k in 1:N
    states[:,k] = pendulum.state
    observations[k] = pendulum.sensor
    step!(pendulum, controls[k])
    torques[k] = pendulum.torque
end

p11 = plot(ylabel="angle")
plot!(tsteps, states[1,:], color="blue", label="state")
scatter!(tsteps, observations, color="black", label="measurements")
p12 = plot(xlabel="time [s]", ylabel="torque")
plot!(tsteps, controls[:], color="red")
plot!(tsteps, torques[:], color="purple")
p10 = plot(p11,p12, layout=grid(2,1, heights=[0.7, 0.3]), size=(900,600))

savefig(p10, "experiments/figures/simsys.png")

## Online system identification

@model function ARXID()

    yk = datavar(Vector{Float64})
    xk = datavar(Vector{Float64})
    μk = datavar(Vector{Float64})
    Λk = datavar(Matrix{Float64})
    αk = datavar(Float64)
    βk = datavar(Float64)

    # Parameter prior
    ζ ~ MvNormalGamma(μk,Λk,αk,βk)

    # Autoregressive likelihood
    yk ~ ARX(xk,ζ)
end

My = 2
Mu = 0
M = My+Mu+1
ybuffer = zeros(My)
ubuffer = zeros(Mu+1)
yk = observations[1]
FE = zeros(N)

for k = 1:N-1

    # Full buffer
    xk = [ybuffer; ubuffer]

    # Extract parameters
    μk,Λk,αk,βk = params(post[:ζ])

    # Update parameter belief
    (results,post) = infer(
        model = ARXID(),
        data = (yk = yk, xk = xk, μk=μk, Λk=Λk, αk=αk, βk=βk),
        free_energy = true,
    )

    # Keep track of free energy
    FE[k] = results.free_energy

    # Update buffers
    yk = observations[k+1]
    ybuffer = backshift(ybuffer,observations[k])
    ubuffer = backshift(ubuffer,torques[k])

end
