
@node LocationScaleT Stochastic [out, ν,μ,σ]


@rule LocationScaleT(:out, Marginalisation) (q_ν::PointMass, q_μ::PointMass, q_σ::PointMass) = begin
    return LocationScaleT(q_ν, q_μ, q_σ)
end

# @rule LocationScaleT(:out, Marginalisation) (ν::PointMass, μ::NormalDistributionsFamily, σ2::GammaDistributionsFamily) = begin
#     return error("todo")
# end 

# @rule LocationScaleT(:ν, Marginalisation) (out::PointMass, μ::NormalDistributionsFamily, σ2::GammaDistributionsFamily) = begin
#     return error("todo")
# end 

# @rule LocationScaleT(:μ, Marginalisation) (out::PointMass, ν::PointMass, σ2::GammaDistributionsFamily) = begin
#     return error("todo")
# end 

# @rule LocationScaleT(:σ, Marginalisation) (out::PointMass, ν:PointMass, μ::NormalDistributionsFamily) = begin
#     return error("todo")
# end 
