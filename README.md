# DifferentiableDFTK

Automatic differentiation for density functional theory in Julia.

This project is part of Google Summer of Code 2021.
https://summerofcode.withgoogle.com/projects/#6407471502983168



## Pull requests

DFTK.jl

ForwardDiff stresses
https://github.com/JuliaMolSim/DFTK.jl/issues/443

**(merged)** Hellmann-Feynman stresses via ForwardDiff and custom rules #476
https://github.com/JuliaMolSim/DFTK.jl/pull/476

**(merged)** Extend ForwardDiff fallback for SVector norm to handle multiple partials #488
https://github.com/JuliaMolSim/DFTK.jl/pull/488

**(merged)** Improve _apply_plan type stability #494
https://github.com/JuliaMolSim/DFTK.jl/pull/494

ForwardDiff scf example
TODO pr to main

Zygote stresses (WIP draft)
TODO pr to main

ChainRules.jl

Add nondiff rules for one ones zero zeros #465
https://github.com/JuliaDiff/ChainRules.jl/pull/465

(Fix #446) Widen _mulsubtrans!! type signature #447
https://github.com/JuliaDiff/ChainRules.jl/pull/447

## Snippets

- NLSolve general implicit differentiation rrule
https://gist.github.com/niklasschmitz/b00223b9e9ba2a37ed09539a264bf423

- linsolve implicit differentiation rrule
- Application example on a small differential equation (linsolve inside nlsolve)

## Issues

- NLSolve

Anderson instability example #273
https://github.com/JuliaNLSolvers/NLsolve.jl/issues/273

participated:
https://github.com/JuliaDiff/ForwardDiff.jl/pull/495

## Workarounds

ForwardDiff
- AbstractFFT rules (based on mcabbotts draft, fixed some bugs & perf penalties; should ideally be upstreamed)
- norm of SVec at zero: custom rule to comply with chainrules convention (and consistency with norm(Vector(x)))
otherwise very pleasant overall and easy to debug


