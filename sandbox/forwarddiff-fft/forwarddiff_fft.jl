using ForwardDiff

#==========================#
# https://github.com/JuliaDiff/ForwardDiff.jl/pull/495/files

using AbstractFFTs
ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) =
    Complex(x.re.value, x.im.value)

ForwardDiff.partials(x::Complex{<:ForwardDiff.Dual}, n::Int) =
    Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.im, n))

ForwardDiff.npartials(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = N
ForwardDiff.npartials(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = N

# AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), region)

    end
end

# rfft only accepts real arrays
AbstractFFTs.plan_rfft(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
    AbstractFFTs.plan_rfft(ForwardDiff.value.(x), region)

for plan in [:plan_irfft, :plan_brfft]  # these take an extra argument, only when complex?
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, d::Integer, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), d, region)

    end
end

for P in [:Plan, :ScaledPlan]  # need ScaledPlan to avoid ambiguities
    @eval begin

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:ForwardDiff.Dual}) =
            _apply_plan(p, x)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            _apply_plan(p, x)

    end
end

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray)
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
    map(xtil, dxtils...) do val, parts...
        Complex(
            ForwardDiff.Dual(real(val), map(real, parts)),
            ForwardDiff.Dual(imag(val), map(imag, parts)),
        )
    end
end


using Test
using FFTW

@testset "fft on Dual" begin
    x1 = ForwardDiff.Dual.(1:4.0, 2:5, 3:6)

    @test ForwardDiff.value.(x1) == 1:4
    @test ForwardDiff.partials.(x1, 1) == 2:5
    
    @test AbstractFFTs.complexfloat(x1)[1] === AbstractFFTs.complexfloat(x1[1]) === ForwardDiff.Dual(1.0, 2.0, 3.0) + 0im
    @test AbstractFFTs.realfloat(x1)[1] === AbstractFFTs.realfloat(x1[1]) === ForwardDiff.Dual(1.0, 2.0, 3.0)
    
    @test fft(x1, 1)[1] isa Complex{<:ForwardDiff.Dual}
    
    @testset "$f" for f in [fft, ifft, rfft, bfft]
        @test ForwardDiff.value.(f(x1)) == f(ForwardDiff.value.(x1))
        @test ForwardDiff.partials.(f(x1), 1) == f(ForwardDiff.partials.(x1, 1))
    end  
end

#==========================#


f(x) = sum(real(rfft(x)))
x = rand(10)
f(x)
ForwardDiff.gradient(f, x)
# ERROR: LoadError: Cannot determine ordering of Dual tags Nothing and ForwardDiff.Tag{typeof(f), Float64}
# Stacktrace:
# [1] ≺(a::Type, b::Type)
#   @ ForwardDiff ~/.julia/packages/ForwardDiff/QOqCN/src/dual.jl:49
# [2] partials
#   @ ~/.julia/packages/ForwardDiff/QOqCN/src/dual.jl:103 [inlined]
# [3] extract_gradient!(#unused#::Type{ForwardDiff.Tag{typeof(f), Float64}}, result::Vector{Float64}, dual::ForwardDiff.Dual{Nothing, Float64, 10})
#   @ ForwardDiff ~/.julia/packages/ForwardDiff/QOqCN/src/gradient.jl:81
# [4] vector_mode_gradient(f::typeof(f), x::Vector{Float64}, cfg::ForwardDiff.GradientConfig{ForwardDiff.Tag{typeof(f), Float64}, Float64, 10, Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(f), Float64}, Float64, 10}}})
#   @ ForwardDiff ~/.julia/packages/ForwardDiff/QOqCN/src/gradient.jl:109
# [5] gradient(f::Function, x::Vector{Float64}, cfg::ForwardDiff.GradientConfig{ForwardDiff.Tag{typeof(f), Float64}, Float64, 10, Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(f), Float64}, Float64, 10}}}, ::Val{true})
#   @ ForwardDiff ~/.julia/packages/ForwardDiff/QOqCN/src/gradient.jl:19
# [6] gradient(f::Function, x::Vector{Float64}, cfg::ForwardDiff.GradientConfig{ForwardDiff.Tag{typeof(f), Float64}, Float64, 10, Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(f), Float64}, Float64, 10}}}) (repeats 2 times)
#   @ ForwardDiff ~/.julia/packages/ForwardDiff/QOqCN/src/gradient.jl:17
# [7] top-level scope
#   @ ~/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/forwarddiff_fft/forwarddiff_fft.jl:103
