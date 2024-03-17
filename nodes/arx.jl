
struct ARX end
@node ARX Stochastic [out, x, ζ]


@rule ARX(:out, Marginalisation) (q_x::PointMass, q_ζ::MvNormalGamma) = begin


    
    return NormalMeanPrecision(out_mean, out_prec)
end

@rule ARX(:x, Marginalisation) (q_out::PointMass, q_ζ::MvNormalGamma) = begin
    
    return NormalMeanPrecision()
end

@rule ARX(:ζ, Marginalisation) (q_out::PointMass, q_x::PointMass) = begin
    
    return MvNormalGamma()
end