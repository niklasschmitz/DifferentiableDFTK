# Solve -u'' + V u + α u^3 = f

using BenchmarkTools
using ChainRulesCore
using FFTW
using FiniteDiff
using IterativeSolvers: gmres # gmres
using KrylovKit: linsolve # linsolve
using LinearAlgebra
using LinearMaps
using NLsolve
using Zygote

N = 100
V(x) = 2 + sin(x)
f(x) = cos(x)
const α = 1.0

# frequencies, avoiding aliasing: eg if N=5, this is 0, 1, 2, -2, 1
ω(i, N) = (i <= div(N,2)+1) ? (i-1) : -(N-i+1)

const x = range(0, 2π, length=N+1)[1:end-1]
const ωs = ω.(1:N, N)
const Vx = V.(x)
const fx = f.(x)

# Zygote specific: need to wrap input u in real(.), otherwise one gets
# implicit promotion to complex numbers in the backwards pass arriving at u.
# real(.) has a custom rrule that ensures real-valued cotangents
# https://github.com/JuliaDiff/ChainRules.jl/blob/65833a19629ee890d30edc96b61bbdbc4e0da72f/src/rulesets/Base/base.jl#L23
lap(u) = real(ifft(ωs.^2 .* fft(real(u)))) # computes u''
precond(u) = real(ifft(fft(real(u)) ./ (1 .+ ωs .^ 2)))

# Goal: find e.g. d(sum(u*)) / dVx via AD

function blackbox_nlsolve_linsolve(Vx)
    residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
    lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
    # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
    function fp_frozen(u_frozen)
        f_inner(u) = precond(lin_op(u, u_frozen))
        linsolve(f_inner, precond(fx), tol=1e-14)[1]
    end
    f_outer(u) = fp_frozen(u) - u
    u_sol2 = nlsolve(f_outer, zeros(N), method=:anderson).zero
    sum(u_sol2)
end

@time blackbox_nlsolve_linsolve(Vx)

# Approach 1: Direct black-box AD with Zygote (unrolling solvers in time)

# Zygote.gradient(blackbox_nlsolve_linsolve, Vx) # ERROR
### the following error was extracted from the vs code debugger (otherwise caught by Zygote in interface2.jl)
# MethodError: no method matching (::var"#3#4")(::Vector{Float64}, ::Vector{Float64})
# Closest candidates are:
#   (::var"#3#4")(::Any) at ~/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/1D/1D.jl:42 (method too new to be called from this world context.)
# Stacktrace:
#   [1] value!!(obj::NLSolversBase.NonDifferentiable{Vector{Float64}, Vector{Float64}}, F::Vector{Float64}, x::Vector{Float64})
#     @ NLSolversBase ~/.julia/packages/NLSolversBase/geyh3/src/interface.jl:166
#   [2] value!!(obj::NLSolversBase.NonDifferentiable{Vector{Float64}, Vector{Float64}}, x::Vector{Float64})
#     @ NLSolversBase ~/.julia/packages/NLSolversBase/geyh3/src/interface.jl:163
#   [3] anderson_(df::NLSolversBase.NonDifferentiable{Vector{Float64}, Vector{Float64}}, initial_x::Vector{Float64}, xtol::Float64, ftol::Float64, iterations::Int64, store_trace::Bool, show_trace::Bool, extended_trace::Bool, beta::Int64, aa_start::Int64, droptol::Float64, cache::NLsolve.AndersonCache{Vector{Float64}, Vector{Float64}, Vector{Vector{Float64}}, Vector{Float64}, Matrix{Float64}, Matrix{Float64}})
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/solvers/anderson.jl:73
#   [4] anderson(df::NLSolversBase.NonDifferentiable{Vector{Float64}, Vector{Float64}}, initial_x::Vector{Float64}, xtol::Float64, ftol::Float64, iterations::Int64, store_trace::Bool, show_trace::Bool, extended_trace::Bool, beta::Int64, aa_start::Int64, droptol::Float64, cache::NLsolve.AndersonCache{Vector{Float64}, Vector{Float64}, Vector{Vector{Float64}}, Vector{Float64}, Matrix{Float64}, Matrix{Float64}})
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/solvers/anderson.jl:203
#   [5] anderson(df::NLSolversBase.NonDifferentiable{Vector{Float64}, Vector{Float64}}, initial_x::Vector{Float64}, xtol::Float64, ftol::Float64, iterations::Int64, store_trace::Bool, show_trace::Bool, extended_trace::Bool, m::Int64, beta::Int64, aa_start::Int64, droptol::Float64)
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/solvers/anderson.jl:188
#   [6] nlsolve(df::NLSolversBase.NonDifferentiable{Vector{Float64}, Vector{Float64}}, initial_x::Vector{Float64}; method::Symbol, xtol::Float64, ftol::Float64, iterations::Int64, store_trace::Bool, show_trace::Bool, extended_trace::Bool, linesearch::Static, linsolve::NLsolve.var"#29#31", factor::Float64, autoscale::Bool, m::Int64, beta::Int64, aa_start::Int64, droptol::Float64)
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/nlsolve/nlsolve.jl:30
#   [7] (::NLsolve.var"#nlsolve##kw")(::NamedTuple{(:method, :show_trace), Tuple{Symbol, Bool}}, ::typeof(nlsolve), df::NLSolversBase.NonDifferentiable{Vector{Float64}, Vector{Float64}}, initial_x::Vector{Float64})
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/nlsolve/nlsolve.jl:18
#   [8] nlsolve(f::var"#3#4", initial_x::Vector{Float64}; method::Symbol, autodiff::Symbol, inplace::Bool, kwargs::Base.Iterators.Pairs{Symbol, Bool, Tuple{Symbol}, NamedTuple{(:show_trace,), Tuple{Bool}}})
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/nlsolve/nlsolve.jl:52
#   [9] (::NLsolve.var"#nlsolve##kw")(::NamedTuple{(:method, :show_trace), Tuple{Symbol, Bool}}, ::typeof(nlsolve), f::var"#3#4", initial_x::Vector{Float64})
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/nlsolve/nlsolve.jl:46
#  [10] top-level scope


#===#
# Approach 2: Add the general rrule of nlsolve

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(nlsolve), f, x0; kwargs...)
    result = nlsolve(f, x0; kwargs...)
    function nlsolve_pullback(Δresult)
        Δx = Δresult[].zero
        x = result.zero
        _, f_pullback = rrule_via_ad(config, f, x)
        JT(v) = f_pullback(v)[2] # w.r.t. x
        # solve JT*Δfx = -Δx
        L = LinearMap(JT, length(x0))
        Δfx = gmres(L, -Δx)
        ∂f = f_pullback(Δfx)[1] # w.r.t. f itself (implicitly closed-over variables)
        return (NoTangent(), ∂f, ZeroTangent())
    end
    return result, nlsolve_pullback
end

# 2.1 Try with nesting NLsolve only (since we already defined its rrule above)

function blackbox_nlsolve_nlsolve(Vx)
    residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
    lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
    # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
    function fp_frozen(u_frozen)
        nlsolve(u -> precond(lin_op(u, u_frozen)) - precond(fx), zeros(N), method=:anderson).zero
    end
    f_outer(u) = fp_frozen(u) - u
    u_sol2 = nlsolve(f_outer, zeros(N), method=:anderson).zero
    sum(u_sol2)
end

let 
    @time blackbox_nlsolve_nlsolve(Vx) # 0.605172 seconds (1.47 M allocations: 105.368 MiB, 10.58% gc time, 98.51% compilation time)
    @time g1 = Zygote.gradient(blackbox_nlsolve_nlsolve, Vx)[1] # 10.130100 seconds (16.95 M allocations: 982.691 MiB, 5.67% gc time, 99.35% compilation time)
    @time g2 = FiniteDiff.finite_difference_gradient(blackbox_nlsolve_nlsolve, Vx) # 3.132538 seconds (4.94 M allocations: 846.075 MiB, 10.13% gc time, 14.61% compilation time)
    @show g1[1:5]
    @show g2[1:5]
    @show sum(abs, g1 - g2) / length(g1) # 0.014476332143572825
    @btime blackbox_nlsolve_nlsolve($x) # 21.492 ms (41011 allocations: 8.33 MiB)
    @btime Zygote.gradient(blackbox_nlsolve_nlsolve, $Vx)[1] # 28.739 ms (116532 allocations: 15.70 MiB)
    nothing
end # looks good


# next step: implement an rrule for linsolve

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(linsolve), A, b; kwargs...)
    result = linsolve(A, b; kwargs...)
    x = result[1]
    function linsolve_pullback(Δresult)
        Δx = Δresult[1]
        _, A_pullback = rrule_via_ad(config, A, x)
        Aᵀ(v) = A_pullback(v)[2]
        b̄, _ = linsolve(Aᵀ, -Δx; kwargs...) # discard convergence info
        Ā = A_pullback(b̄)[1]  # w.r.t. A itself (implicitly closed-over variables)
        return (NoTangent(), Ā, b̄)
    end
    return result, linsolve_pullback
end

let
    @time blackbox_nlsolve_linsolve(Vx) # 0.636815 seconds (1.73 M allocations: 108.934 MiB, 7.03% gc time, 98.51% compilation time)
    @time g1 = Zygote.gradient(blackbox_nlsolve_linsolve, Vx)[1] # 4.534159 seconds (12.70 M allocations: 809.877 MiB, 6.34% gc time, 92.63% compilation time)
    @time g2 = FiniteDiff.finite_difference_gradient(blackbox_nlsolve_linsolve, Vx) # 2.504600 seconds (5.37 M allocations: 791.759 MiB, 7.66% gc time, 6.94% compilation time)
    @show g1[1:5]
    @show g2[1:5]
    @show sum(abs, g1 - g2) / length(g1) # 1.8373434323529963e-6
    @btime blackbox_nlsolve_linsolve($x) # 12.168 ms (37263 allocations: 5.48 MiB)
    @btime Zygote.gradient(blackbox_nlsolve_linsolve, $Vx)[1] # 293.429 ms (1954880 allocations: 132.05 MiB)
    nothing
end
# works! TODO: figure out speed improvement

# TODO alternatively without KrylovKit: implement rrule for IterativeSolvers.gmres

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(gmres), A, b; kwargs...)
    x = gmres(A, b; kwargs...)
    function linsolve_pullback(Δx)
        _, A_pullback = rrule_via_ad(config, A, x)
        Aᵀ(v) = A_pullback(v)[2]
        ∂b = gmres(Aᵀ, -Δx; kwargs...)
        ∂A = A_pullback(∂b)[1]  # w.r.t. A itself (implicitly closed-over variables)
        return (NoTangent(), ∂A, ∂b)
    end
    return x, linsolve_pullback
end

# function blackbox_nlsolve_gmres(Vx)
#     residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
#     lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
#     # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
#     function fp_frozen(u_frozen)
#         L = LinearMap(u -> precond(lin_op(u, u_frozen)), length(precond(fx)))
#         gmres(L, precond(fx))
#     end
#     u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson).zero
#     sum(u_sol)
# end

# let 
#     @time blackbox_nlsolve_gmres(Vx) # 0.636815 seconds (1.73 M allocations: 108.934 MiB, 7.03% gc time, 98.51% compilation time)
#     @time g1 = Zygote.gradient(blackbox_nlsolve_gmres, Vx)[1]
# end
