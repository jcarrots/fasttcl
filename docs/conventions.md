# Physical and numerical conventions

The Hamiltonian is `H = H_S + H_B + lambda A tensor B`, with hbar=1. `A` is
the unscaled Hermitian coupling supplied to `prepare_model`. The bath input
is `C_B(t)=Tr[B(t)B(0)rho_B]`, independent of lambda. The physical correlation
is `lambda^2 C_B(t)`.

`coupling_strength` means lambda; supply it separately from `A` and `C_B`.

The supplied zero-temperature Ohmic bath has
`J_B(w)=w exp(-w/omega_c)` and
`C_B(t)=(1/2) integral_0^infinity J_B(w) exp(-i*w*t) dw`
`=omega_c^2/[2(1+i*omega_c*t)^2]`. Other conventions may place the factor 1/2
inside the spectral density. Custom correlations are used with the normalization you supply.

`generator_series` returns energy-basis **Schrodinger-picture** arrays.
Columns of `model.eigenvectors` are energy eigenvectors in the input basis.
Operators use column-major vectorization: `vec(rho)[i+d*j]=rho[i,j]`.
Generator entries have axes `(time, output_vec_index, input_vec_index)`.

`series.free` is `-i[H_S,.]`. `series.corrections[2]`, `[4]`, and `[6]` are
the physical contributions, including lambda to the corresponding power.
`series.cumulative(4)` is free + correction2 + correction4, for example.

Define `omega_ij=E_i-E_j` and
`Gamma_ij(t)=integral_0^t C_B(s) exp(+i*omega_ij*s) ds`.
Gamma uses cumulative composite trapezoids, with an exact zero at t=0.
The TCL2 filtered operator uses the transposed frequency:
`F_ij(t)=A_ij Gamma_ji(t)`. It includes cross-frequency/nonsecular terms.
TCL4 uses composite-trapezoid memory integrals in its F/C/R reduction, including
the endpoint corrections in the FFT formulas. TCL6 retains the published
discrete endpoint treatment and HR001-HR124 labels.

`generator_series(..., dt=h, n_steps=N)` evaluates at `0,h,...,N*h`.
`solve(..., dt=h, n_steps=N)` evaluates all generators at
`0,h/2,...,N*h` and advances classical RK4 to `0,h,...,N*h`.
The middle RK4 stages use the calculated midpoint generator. Returned
density matrices are transformed back to the input laboratory basis.

For sampled correlations, pass `SampledBath(values, dt)` or the exact array
required by the called function. `solve` needs `2*N+1` half-step samples.
Negative times use `C_B(-t)=conj(C_B(t))`; no interpolation or extrapolation is
performed. The bath must satisfy the physical stationary-Hermitian assumptions.

The solver propagates the density matrix without renormalization or positivity
corrections. A finite TCL truncation does not guarantee positivity. Check
time-step convergence and changes between perturbative orders for each model.
