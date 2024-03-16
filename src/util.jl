using LinearAlgebra
using DSP


function angle2pos(z; l=1.0)
    "Map angle to Cartesian position"
    return (l*sin(z), -l*cos(z))
end

function lowpass(x::Vector; order=1, Wn=1.0, fs=1.0)
    "Extract AR coefficients based on low-pass Butterworth filter"

    dfilter = digitalfilter(Lowpass(Wn; fs=fs), Butterworth(order))
    
    tf = convert(PolynomialRatio, dfilter)
    b = coefb(tf)
    a = coefa(tf)

    return filt(b,a, x), a,b
end

function extract(d::Dict)
    "Define variables key = value for all elements of Dict."
    expr = quote end
    for (k, v) in d
        push!(expr.args, :($(Symbol(k)) = $v))
    end
    eval(expr)
    return
end