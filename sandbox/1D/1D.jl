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

# 2.1 Try with NLsolve only (since we already defined its rrule above)

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


# next step: implement an adjoint rule for linsolve or gmres

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(linsolve), A, b; kwargs...)
    result = linsolve(A, b; kwargs...)
    x = result[1]
    function linsolve_pullback(Δresult)
        Δx = Δresult[1]
        _, A_pullback = rrule_via_ad(config, A, x)
        Aᵀ(v) = A_pullback(v)[2]
        b̄, _ = linsolve(Aᵀ, Δx; kwargs...) # discard convergence info
        Ā = (-b̄) ⊗ x'
        return (NoTangent(), Ā, b̄)
    end
    return (result, linsolve_pullback)
end

## check if the linsolve rrule works on a small linear system
a = let
    x = rand(10, 100)
    x * x' + I
end
b = rand(10)

linsolve(x->a*x, b)
out, back = rrule(Zygote.ZygoteRuleConfig(), linsolve, x->a*x, b)
df = unthunk(back(out)[2])
df(out[1])
back(out)

out2, back2 = Zygote.pullback(A -> A \ b, a)
back2(out2)
out2

out[1] ≈ out2
back2(out2)
back2(out[1])[1]*out[1]
back2(out[1])[1]*out[1] ≈ df(out[1])

# ^seems to work
# TODO now try on the ode (use linsolve inside fp_frozen again)

let 
    blackbox = Vx -> begin
        residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
        lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
        # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
        function fp_frozen(u_frozen)
            result = linsolve(u -> precond(lin_op(u, u_frozen)), precond(fx), tol=1e-14)
            @show result
            result[1]
        end
        u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson).zero
        sum(u_sol)
    end
    # @time blackbox(Vx)
    @time g1 = real(Zygote.gradient(blackbox, Vx)[1]) # LoadError: Need an adjoint for constructor var"#174#180"{Vector{Float64}, var"#lin_op#178"{Vector{Float64}}}. Gradient is of type LinearMaps.KroneckerMap{Float64, Tuple{LinearMaps.WrappedMap{Float64, Vector{Float64}}, LinearMaps.WrappedMap{Float64, Adjoint{Float64, Vector{Float64}}}}}
    # g2 = ForwardDiff.gradient(blackbox, Vx)
    # @time g3 = FiniteDiff.finite_difference_gradient(blackbox, Vx)
    # @show g1[1:5]
    # @show g3[1:5]
    # @show sum(abs, g1 - g3) / length(g1)
    # @btime $blackbox($x)
    # @btime real(Zygote.gradient($blackbox, $Vx)[1])
    # nothing
end
# ERROR: LoadError: Need an adjoint for constructor var"#174#180"{Vector{Float64}, var"#lin_op#178"{Vector{Float64}}}. Gradient is of type LinearMaps.KroneckerMap{Float64, Tuple{LinearMaps.WrappedMap{Float64, Vector{Float64}}, LinearMaps.WrappedMap{Float64, Adjoint{Float64, Vector{Float64}}}}}
# Stacktrace:
#   [1] error(s::String)
#     @ Base ./error.jl:33
#   [2] (::Zygote.Jnew{var"#174#180"{Vector{Float64}, var"#lin_op#178"{Vector{Float64}}}, Nothing, false})(Δ::LinearMaps.KroneckerMap{Float64, Tuple{LinearMaps.WrappedMap{Float64, Vector{Float64}}, LinearMaps.WrappedMap{Float64, Adjoint{Float64, Vector{Float64}}}}})
#     @ Zygote ~/.julia/packages/Zygote/0da6K/src/lib/lib.jl:323
#   [3] (::Zygote.var"#1748#back#209"{Zygote.Jnew{var"#174#180"{Vector{Float64}, var"#lin_op#178"{Vector{Float64}}}, Nothing, false}})(Δ::LinearMaps.KroneckerMap{Float64, Tuple{LinearMaps.WrappedMap{Float64, Vector{Float64}}, LinearMaps.WrappedMap{Float64, Adjoint{Float64, Vector{Float64}}}}})
#     @ Zygote ~/.julia/packages/ZygoteRules/OjfTt/src/adjoint.jl:59
#   [4] Pullback
#     @ ~/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/1D/1D.jl:199 [inlined]
#   [5] (::typeof(∂(λ)))(Δ::SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ Zygote ~/.julia/packages/Zygote/0da6K/src/compiler/interface2.jl:0
#   [6] Pullback
#     @ ~/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/1D/1D.jl:203 [inlined]
#   [7] (::typeof(∂(λ)))(Δ::SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ Zygote ~/.julia/packages/Zygote/0da6K/src/compiler/interface2.jl:0
#   [8] (::Zygote.var"#ad_pullback#41"{var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Tuple{Vector{Float64}}, typeof(∂(λ))})(Δ::SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ Zygote ~/.julia/packages/Zygote/0da6K/src/compiler/chainrules.jl:131
#   [9] JT
#     @ ~/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/1D/1D.jl:80 [inlined]
#  [10] _unsafe_mul!(y::SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}, A::LinearMaps.FunctionMap{Float64, var"#JT#20"{Zygote.var"#ad_pullback#41"{var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Tuple{Vector{Float64}}, typeof(∂(λ))}}, Nothing}, x::SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ LinearMaps ~/.julia/packages/LinearMaps/r6i00/src/functionmap.jl:109
#  [11] mul!
#     @ ~/.julia/packages/LinearMaps/r6i00/src/LinearMaps.jl:147 [inlined]
#  [12] expand!(arnoldi::IterativeSolvers.ArnoldiDecomp{Float64, LinearMaps.FunctionMap{Float64, var"#JT#20"{Zygote.var"#ad_pullback#41"{var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Tuple{Vector{Float64}}, typeof(∂(λ))}}, Nothing}}, Pl::Identity, Pr::Identity, k::Int64, Ax::Vector{Float64})
#     @ IterativeSolvers ~/.julia/packages/IterativeSolvers/xIVNp/src/gmres.jl:287
#  [13] iterate(g::IterativeSolvers.GMRESIterable{Identity, Identity, Vector{Float64}, FillArrays.Fill{Float64, 1, Tuple{Base.OneTo{Int64}}}, Vector{Float64}, IterativeSolvers.ArnoldiDecomp{Float64, LinearMaps.FunctionMap{Float64, var"#JT#20"{Zygote.var"#ad_pullback#41"{var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Tuple{Vector{Float64}}, typeof(∂(λ))}}, Nothing}}, IterativeSolvers.Residual{Float64, Float64}, Float64, IterativeSolvers.ModifiedGramSchmidt}, iteration::Int64)
#     @ IterativeSolvers ~/.julia/packages/IterativeSolvers/xIVNp/src/gmres.jl:64
#  [14] iterate
#     @ ~/.julia/packages/IterativeSolvers/xIVNp/src/gmres.jl:59 [inlined]
#  [15] iterate
#     @ ./iterators.jl:159 [inlined]
#  [16] iterate
#     @ ./iterators.jl:158 [inlined]
#  [17] gmres!(x::Vector{Float64}, A::LinearMaps.FunctionMap{Float64, var"#JT#20"{Zygote.var"#ad_pullback#41"{var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Tuple{Vector{Float64}}, typeof(∂(λ))}}, Nothing}, b::FillArrays.Fill{Float64, 1, Tuple{Base.OneTo{Int64}}}; Pl::Identity, Pr::Identity, abstol::Float64, reltol::Float64, restart::Int64, maxiter::Int64, log::Bool, initially_zero::Bool, verbose::Bool, orth_meth::IterativeSolvers.ModifiedGramSchmidt)
#     @ IterativeSolvers ~/.julia/packages/IterativeSolvers/xIVNp/src/gmres.jl:207
#  [18] gmres(A::LinearMaps.FunctionMap{Float64, var"#JT#20"{Zygote.var"#ad_pullback#41"{var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Tuple{Vector{Float64}}, typeof(∂(λ))}}, Nothing}, b::FillArrays.Fill{Float64, 1, Tuple{Base.OneTo{Int64}}}; kwargs::Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
#     @ IterativeSolvers ~/.julia/packages/IterativeSolvers/xIVNp/src/gmres.jl:143
#  [19] gmres(A::LinearMaps.FunctionMap{Float64, var"#JT#20"{Zygote.var"#ad_pullback#41"{var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Tuple{Vector{Float64}}, typeof(∂(λ))}}, Nothing}, b::FillArrays.Fill{Float64, 1, Tuple{Base.OneTo{Int64}}})
#     @ IterativeSolvers ~/.julia/packages/IterativeSolvers/xIVNp/src/gmres.jl:143
#  [20] (::var"#nlsolve_pullback#19"{Zygote.ZygoteRuleConfig{Zygote.Context}, var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Vector{Float64}, NLsolve.SolverResults{Float64, Float64, Vector{Float64}, Vector{Float64}}})(Δresult::Base.RefValue{Any})
#     @ Main ~/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/1D/1D.jl:83
#  [21] ZBack
#     @ ~/.julia/packages/Zygote/0da6K/src/compiler/chainrules.jl:91 [inlined]
#  [22] (::Zygote.var"#kw_zpullback#40"{var"#nlsolve_pullback#19"{Zygote.ZygoteRuleConfig{Zygote.Context}, var"#175#181"{var"#fp_frozen#179"{var"#lin_op#178"{Vector{Float64}}}}, Vector{Float64}, NLsolve.SolverResults{Float64, Float64, Vector{Float64}, Vector{Float64}}}})(dy::Base.RefValue{Any})
#     @ Zygote ~/.julia/packages/Zygote/0da6K/src/compiler/chainrules.jl:117
#  [23] Pullback
#     @ ~/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/1D/1D.jl:203 [inlined]
#  [24] (::typeof(∂(#173)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/0da6K/src/compiler/interface2.jl:0
#  [25] (::Zygote.var"#46#47"{typeof(∂(#173))})(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/0da6K/src/compiler/interface.jl:41
#  [26] gradient(f::Function, args::Vector{Float64})
#     @ Zygote ~/.julia/packages/Zygote/0da6K/src/compiler/interface.jl:59
#  [27] macro expansion
#     @ timing.jl:210 [inlined]
#  [28] top-level scope
#     @ ~/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/1D/1D.jl:207
# in expression starting at /home/niku/Documents/cloud/sandbox/julia/dftk/DifferentiableDFTK/sandbox/1D/1D.jl:193

# using ChainRulesCore
# function ChainRulesCore.rrule(::Type{LinearMap}, f, M::Int)
#     L = LinearMap(f, M)
#     LinearMap_pullback(ΔL) = (NoTangent(), ΔL.f, NoTangent())
#     return L, LinearMap_pullback
# end
# Zygote.refresh()
