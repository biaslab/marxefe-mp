
struct MARX end

@node MARX Stochastic [out, outprev1, outprev2, in, inprev1, inprev2, Φ]

## Average energy

# TODO!
# Average energy of MARX node
#    η_t = ν - Dy + 1
#    μ_t = M'*x_t
#    Σ_t = 1/(ν-Dy+1)*Ω*(1 + x_t'*inv(Λ)*x_t)
#    Ψ = inv(Σ_t)
#    -1/2*(agent.Dy*log(η*π) -logdet(Ψ) - 2*logmultigamma(agent.Dy, (η+agent.Dy)/2) + 2*logmultigamma(agent.Dy, (η+agent.Dy-1)/2) + (η+agent.Dy)*log(1 + 1/η*(y-μ)'*Ψ*(y-μ)) )