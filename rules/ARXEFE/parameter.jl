
@rule ARXEFE(:ζ, Marginalisation) (q_out::PointMass,
                                   q_outprev1::PointMass, 
                                   q_outprev2::PointMass, 
                                   q_in::PointMass, 
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass) = begin

    yk = mean(q_out)                                
    xk = [mean(q_outprev1), mean(q_outprev2), mean(q_in), mean(q_inprev1), mean(q_inprev2)]
    M  = length(xk)

    μ_ = inv(xk*xk' + 1e-8*I(M))*(xk*yk)
    Λ_ = xk*xk' + 1e-8*I(M)
    α_ = -M/2 + 3/2.
    β_ = 0.0

    return MvNormalGamma(μ_, Λ_, α_, β_)
end

@rule ARXEFE(:ζ, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily,
                                   q_outprev1::PointMass, 
                                   q_outprev2::PointMass, 
                                   m_in::UnivariateGaussianDistributionsFamily, 
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass) = begin

    return Uninformative()
end

@rule ARXEFE(:ζ, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily,
                                   q_outprev1::UnivariateGaussianDistributionsFamily, 
                                   q_outprev2::PointMass, 
                                   m_in::UnivariateGaussianDistributionsFamily, 
                                   q_inprev1::UnivariateGaussianDistributionsFamily, 
                                   q_inprev2::PointMass) = begin

    return Uninformative()
end

@rule ARXEFE(:ζ, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily,
                                   q_outprev1::UnivariateGaussianDistributionsFamily, 
                                   q_outprev2::UnivariateGaussianDistributionsFamily, 
                                   m_in::UnivariateGaussianDistributionsFamily, 
                                   q_inprev1::UnivariateGaussianDistributionsFamily, 
                                   q_inprev2::UnivariateGaussianDistributionsFamily) = begin

    return Uninformative()
end

@rule ARXEFE(:ζ, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily,
                                   q_outprev1::PointMass, 
                                   q_outprev2::PointMass, 
                                   m_in::Uniform, 
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass) = begin

    return Uninformative()
end