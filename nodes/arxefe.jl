
struct ARXEFE end
@node ARXEFE Stochastic [mstar, vstar, x, u, ζ]


@rule ARXEFE(:u, Marginalisation) (q_mstar::PointMass, q_vstar::PointMass, q_x::PointMass, q_ζ::MvNormalGamma) = begin
    
    return 
end

@rule ARXEFE(:x, Marginalisation) (q_mstar::PointMass, q_vstar::PointMass, q_u::PointMass, q_ζ::MvNormalGamma) = begin
    return Uninformative()
end

@rule ARXEFE(:ζ, Marginalisation) (q_mstar::PointMass, q_vstar::PointMass, q_x::PointMass, q_u::PointMass) = begin
    return Uninformative()
end