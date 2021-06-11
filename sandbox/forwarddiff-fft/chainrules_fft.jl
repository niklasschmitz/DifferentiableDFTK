using AbstractFFTs
using ChainRulesCore

# ref: Zygote @adjoint rules for AbstractFFTs
# https://github.com/FluxML/Zygote.jl/blob/12f5c1d75eeaa8c7a818f2db7f8d082956c00cac/src/lib/array.jl#L759

function ChainRulesCore.frule((_, Δxs), ::typeof(fft), xs)
    ys = fft(xs)
    ∂ys = fft(Δxs)
    return ys, ∂ys
end

function ChainRulesCore.rrule(::typeof(fft), xs)   
    ys = fft(xs)
    function fft_pullback(Δys)
        ∂xs = bfft(Δys)
        return (NoTangent(), ∂xs)
    end
    return ys, fft_pullback
end

# TODO add more

#===#
using ChainRulesTestUtils
using FFTW
x = rand(5)
test_frule(fft, x)
test_rrule(fft, x) # TODO
