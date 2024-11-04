
@rule MvLocationScaleT(:out, Marginalisation) (q_ν::PointMass, q_μ::PointMass, q_σ::PointMass) = begin
    return MvLocationScaleT(q_ν, q_μ, q_σ)
end