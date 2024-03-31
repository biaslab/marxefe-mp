
@node MvLocationScaleT Stochastic [out, ν,μ,σ]


@rule MvLocationScaleT(:out, Marginalisation) (q_ν::PointMass, q_μ::PointMass, q_σ::PointMass) = begin
    return MvLocationScaleT(q_ν, q_μ, q_σ)
end

# @rule MvLocationScaleT(:out, Marginalisation) (ν::PointMass, μ::NormalDistributionsFamily, Σ::GammaDistributionsFamily) = begin
#     return error("todo")
# end 

# @rule MvLocationScaleT(:ν, Marginalisation) (out::PointMass, μ::NormalDistributionsFamily, Σ::GammaDistributionsFamily) = begin
#     return error("todo")
# end 

# @rule MvLocationScaleT(:μ, Marginalisation) (out::PointMass, ν::PointMass, Σ::GammaDistributionsFamily) = begin
#     return error("todo")
# end 

# @rule MvLocationScaleT(:Σ, Marginalisation) (out::PointMass, ν:PointMass, μ::NormalDistributionsFamily) = begin
#     return error("todo")
# end 