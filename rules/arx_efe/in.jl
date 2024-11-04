
@rule ARXEFE(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily,
                                    q_outprev1::PointMass, 
                                    q_outprev2::PointMass, 
                                    q_inprev1::PointMass, 
                                    q_inprev2::PointMass, 
                                    m_ζ::MvNormalGamma) = begin

    μ, Λ, α, β = params(m_ζ)
    m_star = mean(m_out)
    v_star = var( m_out)

    function f(u)
        x_t = [mean(q_outprev1), mean(q_outprev2), u, mean(q_inprev1), mean(q_outprev2)]
        return ( s2_t + (m_t - m_star)^2 ) / ( 2v_star ) - 1/2*log(1 + x_t'*inv(Λ)*x_t)
    end 

    return ContinuousUnivariateLogPdf(f)
end