
struct LocationScaleT end
@node LocationScaleT Stochastic [out, ν,μ,σ]


@rule LocationScaleT(:out, Marginalisation) (q_ν::PointMass, q_μ::PointMass, q_σ::PointMass) = begin
    return LocationScaleT(q_ν, q_μ, q_σ)
end

