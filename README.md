# DifferentiableDFTK

Automatic differentiation for density functional theory in Julia.

This project is part of Google Summer of Code 2021.
https://summerofcode.withgoogle.com/projects/#6407471502983168



# Pull requests

## DFTK.jl

Main Project Goal: Hellmann-Feynman stresses via ForwardDiff      
https://github.com/JuliaMolSim/DFTK.jl/issues/443

Main PR implementing the needed infrastructure for ForwardDiff compatibility:  
**(merged)** Hellmann-Feynman stresses via ForwardDiff and custom rules #476  
https://github.com/JuliaMolSim/DFTK.jl/pull/476

Follow-up PRs:  
**(merged)** Extend ForwardDiff fallback for SVector norm to handle multiple partials #488  
https://github.com/JuliaMolSim/DFTK.jl/pull/488

**(merged)** Improve _apply_plan type stability #494  
https://github.com/JuliaMolSim/DFTK.jl/pull/494

**How to use**: To the end user, the new feature of calculating stresses of a converged solution can now be accessed by calling `stresses = compute_stresses(scfres)`. For a complete example, see the testcases in https://github.com/JuliaMolSim/DFTK.jl/blob/master/test/stresses.jl.  

Other stretch goals:

(draft) ForwardDiff example of implicit differentiation beyond the Hellman-Feynman theorem, to calculate a dipole moment.

(draft) Hellmann-Feynman derivatives using ChainRules and Zygote #519  
https://github.com/JuliaMolSim/DFTK.jl/pull/519

## ChainRules.jl

**(merged)** Add nondiff rules for one ones zero zeros #465  
https://github.com/JuliaDiff/ChainRules.jl/pull/465

**(merged)** (Fix #446) Widen _mulsubtrans!! type signature #447  
https://github.com/JuliaDiff/ChainRules.jl/pull/447

## Snippets

- NLsolve general implicit differentiation rrule
https://gist.github.com/niklasschmitz/b00223b9e9ba2a37ed09539a264bf423

- linsolve implicit differentiation rrule https://github.com/niklasschmitz/DifferentiableDFTK/blob/main/sandbox/1D/1D.jl#L128-L140

- Application example on a small differential equation (linsolve inside nlsolve) https://github.com/niklasschmitz/DifferentiableDFTK/blob/main/sandbox/1D/1D.jl

## Issues

NLSolve.jl: Anderson instability example #273
https://github.com/JuliaNLSolvers/NLsolve.jl/issues/273


## Workarounds

ForwardDiff
- AbstractFFT rules (based on mcabbotts draft https://github.com/JuliaDiff/ForwardDiff.jl/pull/495, fixed some bugs & perf penalties; should ideally be upstreamed)
- norm of SVec at zero: custom rule to comply with chainrules convention (and consistency with norm(Vector(x)))
otherwise very pleasant overall and easy to debug


