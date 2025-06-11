using LinearAlgebra


function angle2pos(z; l=1.0)
    "Map angle to Cartesian position"
    return (l*sin(z), -l*cos(z))
end

function backshift(x::AbstractVector, a::Number)
    "Shift elements down and add element"

    N = size(x,1)

    # Shift operator
    S = Tridiagonal(ones(N-1), zeros(N), zeros(N-1))

    # Basis vector
    e = [1.0; zeros(N-1)]

    return S*x + e*a
end

function backshift(M::AbstractMatrix, a::Number)
    return diagm(backshift(diag(M), a))
end

function backshift(x::AbstractMatrix, a::Vector)
    return [a x[:,1:end-1]]
end

function proj2psd(S::AbstractMatrix)
    L,V = eigen(S)
    S = V*diagm(max.(1e-8,L))*V'
    return (S+S')/2
end

function sqrtm(M::AbstractMatrix)
    "Square root of matrix"

    if size(M) == (2,2)
        "https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix"

        A,C,B,D = M

        # Determinant
        δ = A*D - B*C
        s = sqrt(δ)

        # Trace
        τ = A+D
        t = sqrt(τ + 2s)

        return 1/t*(M+s*Matrix{eltype(M)}(I,2,2))
    else
        "Babylonian method"

        Xk = Matrix{eltype(M)}(I,size(M))
        Xm = zeros(eltype(M), size(M))

        while sum(abs.(Xk[:] .- Xm[:])) > 1e-3
            Xm = Xk
            Xk = (Xm + M/Xm)/2.0
        end
        return Xk
    end
end

function tmean(x::AbstractArray; tr::Real=0.2)
    "Taken from RobustStats.jl"
    
    if tr < 0 || tr > 0.5
        error("tr cannot be smaller than 0 or larger than 0.5")
    elseif tr == 0
        return mean(x)
    elseif tr == .5
        return median!(x)
    else
        n   = length(x)
        lo  = floor(Int64, n*tr)+1
        hi  = n+1-lo
        return mean(sort!(x)[lo:hi])
    end
end

function winval(x::AbstractArray; tr::Real=0.2)
    "Taken from RobustStats.jl"

    n = length(x)
    xcopy   = sort(x)
    ibot    = floor(Int64, tr*n)+1
    itop    = n-ibot+1
    xbot, xtop = xcopy[ibot], xcopy[itop]
    return  [x[i]<=xbot ? xbot : (x[i]>=xtop ? xtop : x[i]) for i=1:n]
end

function winvar(x::AbstractArray; tr=0.2) 
    "Taken from RobustStats.jl"

    return var(winval(x, tr=tr))
end

function trimse(x::AbstractArray; tr::Real=0.2)
    "Taken from RobustStats.jl"

    return sqrt(winvar(x,tr=tr))/((1-2tr)*sqrt(length(x)))
end