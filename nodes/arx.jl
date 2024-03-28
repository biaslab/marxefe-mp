
struct ARX end
@node ARX Stochastic [out, x, ζ]


@rule ARX(:out, Marginalisation) (q_x::PointMass, q_ζ::MvNormalGamma) = begin

    mx = mean(q_x)
    μ,Λ,α,β = params(q_ζ)
    
    return NormalMeanPrecision(μ'*mx, mx'*inv(α/β*Λ)*mx + β/α)
end

@rule ARX(:x, Marginalisation) (q_out::PointMass, q_ζ::MvNormalGamma) = begin
    
    return NormalMeanPrecision(0.,1.)
end

@rule ARX(:ζ, Marginalisation) (q_out::PointMass, q_x::PointMass) = begin

    my = mean(q_out)
    mx = mean(q_x)
    D  = length(mx)

    imxmx = inv(mx*mx' + 1e-8diagm(ones(D)))

    μ = imxmx*(mx*my)
    Λ = mx*mx'
    α = 1.0
    β = 1/2*(my^2 - (my*mx')*imxmx*(mx*my)) + 1e-8
    
    return MvNormalGamma(μ,Λ,α,β)
end