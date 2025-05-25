
@rule MatrixNormalWishart(:out, Marginalisation) (q_M::PointMass, 
                                                  q_Λ::PointMass, 
                                                  q_Ω::PointMass,
                                                  q_ν::PointMass) = begin
    return MatrixNormalWishart(mean(q_M),mean(q_Λ),mean(q_Ω),mean(q_ν))
end