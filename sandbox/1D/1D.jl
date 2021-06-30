# Solve -u'' + V u + α u^3 = f

using BenchmarkTools
using ChainRulesCore
using FFTW
using FiniteDiff
using IterativeSolvers # gmres
using KrylovKit # linsolve
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

residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u

# we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
function fp_frozen(u_frozen)
    linsolve(u -> precond(lin_op(u, u_frozen)), precond(fx), tol=1e-14)[1]
end

u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson, show_trace=true).zero
maximum(abs, residual(u_sol))

## directly trying to solve without splitting doesn't work out numerically
# nlsolve(u -> residual(u), zeros(N), method = :anderson, show_trace=true).zero

#===#
# TODO: find e.g. d(sum(u*)) / dVx via AD

#===#
# Approach 1: Direct black-box AD with Zygote (unrolling solvers in time)

let 
    blackbox = Vx -> begin
        residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
        lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
        # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
        function fp_frozen(u_frozen)
            linsolve(u -> precond(lin_op(u, u_frozen)), precond(fx), tol=1e-14)[1]
        end
        u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson, show_trace=true).zero
        sum(u_sol)
    end
    @time blackbox(Vx)
    Zygote.gradient(blackbox, Vx) # ERROR: LoadError: Mutating arrays is not supported
end

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

let 
    blackbox = Vx -> begin
        residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
        lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
        # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
        function fp_frozen(u_frozen)
            nlsolve(u -> precond(lin_op(u, u_frozen)) - precond(fx), zeros(N), method = :anderson).zero
        end
        u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson).zero
        sum(u_sol)
    end
    @time blackbox(Vx) # 0.605172 seconds (1.47 M allocations: 105.368 MiB, 10.58% gc time, 98.51% compilation time)
    @time g1 = Zygote.gradient(blackbox, Vx)[1] # 10.130100 seconds (16.95 M allocations: 982.691 MiB, 5.67% gc time, 99.35% compilation time)
    # g2 = ForwardDiff.gradient(blackbox, Vx)
    @time g3 = FiniteDiff.finite_difference_gradient(blackbox, Vx) # 3.132538 seconds (4.94 M allocations: 846.075 MiB, 10.13% gc time, 14.61% compilation time)
    @show g1[1:5]
    @show g3[1:5]
    @show sum(abs, g1 - g3) / length(g1) # 0.014476332143572825
    @btime $blackbox($x) # 21.492 ms (41011 allocations: 8.33 MiB)
    @btime Zygote.gradient($blackbox, $Vx)[1] # 28.739 ms (116532 allocations: 15.70 MiB)
    nothing
end # looks promising


# next step: implement an rrule for linsolve or gmres

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
    return (result, linsolve_pullback)
end

let 
    blackbox = Vx -> begin
        residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
        lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
        # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
        function fp_frozen(u_frozen)
            linsolve(u -> precond(lin_op(u, u_frozen)), precond(fx), tol=1e-14)[1]
        end
        u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson).zero
        sum(u_sol)
    end
    @time blackbox(Vx) # 0.636815 seconds (1.73 M allocations: 108.934 MiB, 7.03% gc time, 98.51% compilation time)
    @time g1 = Zygote.gradient(blackbox, Vx)[1] # 4.534159 seconds (12.70 M allocations: 809.877 MiB, 6.34% gc time, 92.63% compilation time)
    @time g2 = FiniteDiff.finite_difference_gradient(blackbox, Vx) # 2.504600 seconds (5.37 M allocations: 791.759 MiB, 7.66% gc time, 6.94% compilation time)
    @show g1[1:5]
    @show g2[1:5]
    @show sum(abs, g1 - g2) / length(g1) # 1.8373434323529963e-6
    @btime $blackbox($x) # 12.168 ms (37263 allocations: 5.48 MiB)
    @btime Zygote.gradient($blackbox, $Vx)[1] # 293.429 ms (1954880 allocations: 132.05 MiB)
    nothing
end
# works! TODO: figure out speed improvement
