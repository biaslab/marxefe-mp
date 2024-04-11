using LinearAlgebra
using Distributions
using SpecialFunctions
using Plots
default(label="", linewidth=2)

### Sampling

# Component distributions
mx,vx = (1.0,2.0)
my,vy = (2.0,3.0)
px = Normal(mx,sqrt(vx))
py = Normal(my,sqrt(vy))

N = 100_000

x = rand(px,N)
y = rand(py,N)
z = x .* y

mdz_hat = median(z)
mz_hat  = mean(z)
vz_hat  = var(z)

plot()
histogram(z, normalize=:pdf)

### Overlay functions

# Bessel function
xr = range(-10,stop=10,step=0.5)
pz(z) = besselk.(0, abs(z))./π
plot!(xr, pz.(xr), lw=3, color="red", xlims=extrema(xr))

# Gaussian with moment matching (EP)
qmz = mx*my
qvz = (vx + mx^2)*(vy + my^2) - mx^2*my^2
qz(z) = pdf(Normal(qmz,sqrt(qvz)),z)
plot!(xr, qz.(xr), lw=3, color="purple", xlims=extrema(xr))

