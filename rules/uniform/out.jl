
@rule Uniform(:out, Marginalisation) (q_a::PointMass, 
                                      q_b::PointMass) = begin

    return Uniform(mean(q_a), mean(q_b))
end

@variational_rule Uniform(:out, Marginalisation) (q_a::PointMass, 
                                                  q_b::PointMass) = begin

    return Uniform(mean(q_a), mean(q_b))
end