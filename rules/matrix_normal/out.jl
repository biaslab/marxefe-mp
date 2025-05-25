
@rule MatrixNormal(:out, Marginalisation) (q_M::PointMass, 
                                           q_U::PointMass, 
                                           q_V::PointMass) = begin
    return MatrixNormal(mean(q_M),mean(q_U),mean(q_V))
end