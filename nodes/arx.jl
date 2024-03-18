
struct ARX end
@node ARX Stochastic [out, x, ζ]


@rule ARX(:out, Marginalisation) (q_x::PointMass, q_ζ::MvNormalGamma) = begin

    mx = mean(q_x)
    μ,Λ,α,β = params(q_ζ)
    
    return NormalMeanPrecision(μ'*mx, mx'*inv(α/β*Λ)*mx + β/α)
end

@rule ARX(:x, Marginalisation) (q_out::PointMass, q_ζ::MvNormalGamma) = begin
    
    return NormalMeanPrecision()
end

@rule ARX(:ζ, Marginalisation) (q_out::PointMass, q_x::PointMass) = begin

    mx = mean(q_x)
    my = mean(q_out)

    μ = inv(mx*mx')*(mx*my)
    Λ = mx*mx'
    α = 1/2
    β = 1/2*(my^2 - (mx*my)'*inv(mx*mx')*(mx*my))
    
    return MvNormalGamma(μ,Λ,α,β)
end