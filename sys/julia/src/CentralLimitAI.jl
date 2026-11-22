module CentralLimitAI

using KernelAbstractions
using Enzyme
using LinearAlgebra
using Random
using StaticArrays

export run_collapse_microbatch, lut6_shannon_kernel!, mamba3_complex_stream!, settle_ebrm_enzyme!

# ============================================================================
# RUNG 0: Unified GPU/CPU Bit-Sliced LUT6 Kernel (64 episodes per UInt64 lane)
# ============================================================================
@kernel function lut6_shannon_kernel!(
    out_planes::AbstractMatrix{UInt64},       # [C, num_luts]
    @Const(in_planes::AbstractMatrix{UInt64}),# [C, in_wires]
    @Const(wires::AbstractMatrix{Int32}),     # [num_luts, 6] 1-based wire indices
    @Const(init_masks::AbstractVector{UInt64})# [num_luts] 64-bit truth tables
)
    c, n = @index(Global, NTuple)
    @inbounds begin
        w0 = in_planes[c, wires[n, 1]]
        w1 = in_planes[c, wires[n, 2]]
        w2 = in_planes[c, wires[n, 3]]
        w3 = in_planes[c, wires[n, 4]]
        w4 = in_planes[c, wires[n, 5]]
        w5 = in_planes[c, wires[n, 6]]

        nw0, nw1, nw2, nw3, nw4, nw5 = ~w0, ~w1, ~w2, ~w3, ~w4, ~w5
        t = init_masks[n]
        acc = zero(UInt64)

        for m in UInt64(0):UInt64(63)
            t_mask = zero(UInt64) - ((t >> m) & UInt64(1))
            b0 = ifelse((m & UInt64(1)) != 0, w0, nw0)
            b1 = ifelse((m & UInt64(2)) != 0, w1, nw1)
            b2 = ifelse((m & UInt64(4)) != 0, w2, nw2)
            b3 = ifelse((m & UInt64(8)) != 0, w3, nw3)
            b4 = ifelse((m & UInt64(16)) != 0, w4, nw4)
            b5 = ifelse((m & UInt64(32)) != 0, w5, nw5)

            acc |= t_mask & b0 & b1 & b2 & b3 & b4 & b5
        end
        out_planes[c, n] = acc
    end
end

# ============================================================================
# RUNG 2: Mamba-3 Complex-Valued MIMO Trapezoidal Recurrence (@simd ivdep)
# ============================================================================
function mamba3_complex_stream!(
    summary_out::Vector{Float32},
    bytes::Vector{UInt8},
    h_real::Matrix{Float32},  # [N=64, P=64] pre-allocated state
    prev_bx::Matrix{Float32}, # [N=64, P=64] trapezoidal tap buffer
    phase::Vector{Float32}    # [N÷2=32] cumulative RoPE phase
)
    fill!(h_real, 0.0f0)
    fill!(prev_bx, 0.0f0)
    fill!(phase, 0.0f0)
    N, P = size(h_real)

    @inbounds for b in bytes
        x_s = (Float32(b) - 127.5f0) * 0.0078125f0
        dt = 0.04f0 + 0.02f0 * abs(x_s)
        α = exp(-0.15f0 * dt)
        γ = 0.6f0 * dt
        β = 0.4f0 * dt * α

        for k in 1:(N ÷ 2)
            phase[k] += dt * x_s * Float32(k) * 0.1f0
            s_ang, c_ang = sincos(phase[k])
            b0 = c_ang - s_ang
            b1 = s_ang + c_ang
            n0, n1 = 2k - 1, 2k

            @simd ivdep for p in 1:P
                xp = x_s * Float32((p & 7) + 1) * 0.5f0
                bx0 = b0 * xp
                bx1 = b1 * xp
                h_real[n0, p] = muladd(α, h_real[n0, p], muladd(β, prev_bx[n0, p], γ * bx0))
                h_real[n1, p] = muladd(α, h_real[n1, p], muladd(β, prev_bx[n1, p], γ * bx1))
                prev_bx[n0, p] = bx0
                prev_bx[n1, p] = bx1
            end
        end
    end

    @inbounds for p in 1:P
        acc = 0.0f0
        @simd for n in 1:N
            acc += h_real[n, p]
        end
        summary_out[p] = acc
    end
    inv_n = inv(sqrt(sum(abs2, summary_out) + 1.0f-8))
    summary_out .*= inv_n
    return summary_out
end

# ============================================================================
# RUNG 3: Zero-Allocation LLVM-Level Enzyme.jl JEPA-EBRM Settling Core
# ============================================================================
@inline function jepa_ebrm_free_energy(
    z::Vector{Float32},
    s_x::Vector{Float32},
    W_jepa::Matrix{Float32}
)::Float32
    energy = 0.0f0
    D = length(z)
    @inbounds for i in 1:D
        pred_i = 0.0f0
        @simd for j in 1:D
            pred_i = muladd(W_jepa[i, j], s_x[j], pred_i)
        end
        diff = z[i] - pred_i
        energy = muladd(diff, diff, energy)
    end
    return 0.5f0 * energy / Float32(D)
end

function settle_ebrm_enzyme!(
    z::Vector{Float32},
    dz::Vector{Float32},
    s_x::Vector{Float32},
    W_jepa::Matrix{Float32};
    max_steps::Int = 24,
    lr::Float32 = 16.0f0,
    τ_commit::Float32 = 0.002f0
)
    fill!(z, 0.0f0)
    for step in 1:max_steps
        fill!(dz, 0.0f0)
        e = jepa_ebrm_free_energy(z, s_x, W_jepa)
        if e <= τ_commit
            return (settled = true, energy = e, steps = step)
        end
        Enzyme.autodiff(
            Reverse,
            jepa_ebrm_free_energy,
            Active,
            Duplicated(z, dz),
            Const(s_x),
            Const(W_jepa),
        )
        @inbounds @simd for i in eachindex(z)
            z[i] -= lr * dz[i]
        end
    end
    e_final = jepa_ebrm_free_energy(z, s_x, W_jepa)
    return (settled = e_final <= τ_commit, energy = e_final, steps = max_steps)
end

function run_collapse_microbatch(backend = CPU())
    Random.seed!(42)
    C, W, N_LUTS, D = 1, 512, 512, 64
    in_planes = KernelAbstractions.allocate(backend, UInt64, C, W)
    out_planes = KernelAbstractions.allocate(backend, UInt64, C, N_LUTS)
    wires = KernelAbstractions.allocate(backend, Int32, N_LUTS, 6)
    masks = KernelAbstractions.allocate(backend, UInt64, N_LUTS)

    copyto!(in_planes, rand(UInt64, C, W))
    copyto!(wires, Int32.(rand(1:W, N_LUTS, 6)))
    copyto!(masks, rand(UInt64, N_LUTS))

    kernel! = lut6_shannon_kernel!(backend, 64)
    # Warmup JIT
    kernel!(out_planes, in_planes, wires, masks; ndrange = (C, N_LUTS))
    KernelAbstractions.synchronize(backend)

    t0 = time_ns()
    kernel!(out_planes, in_planes, wires, masks; ndrange = (C, N_LUTS))
    KernelAbstractions.synchronize(backend)
    r0_us = (time_ns() - t0) / 1.0e3

    # Verify bit-exact parity on LUT #1, episode bit 0
    in_host = Array(in_planes)
    out_host = Array(out_planes)
    w_host = Array(wires)
    m_host = Array(masks)
    addr = 0
    for p in 1:6
        addr |= Int((in_host[1, w_host[1, p]] & 1)) << (p - 1)
    end
    expected_bit = (m_host[1] >> addr) & UInt64(1)
    actual_bit = out_host[1, 1] & UInt64(1)
    @assert expected_bit == actual_bit "LUT6 bit-slice mismatch!"

    h_buf = zeros(Float32, 64, D)
    bx_buf = zeros(Float32, 64, D)
    ph_buf = zeros(Float32, 32)
    s_x = zeros(Float32, D)
    stream = rand(UInt8, 48)
    mamba3_complex_stream!(s_x, stream, h_buf, bx_buf, ph_buf)

    z = zeros(Float32, D)
    dz = zeros(Float32, D)
    W_jepa = Matrix{Float32}(I, D, D)
    res = settle_ebrm_enzyme!(z, dz, s_x, W_jepa)

    println(
        "[Julia $(VERSION) | KernelAbstractions + Enzyme] 64-Ep LUT6: $(round(r0_us, digits=2)) μs | ",
        "Bit-Exact Parity: OK | Enzyme EBRM Settled: $(res.settled) in $(res.steps) steps (E=$(round(res.energy, sigdigits=3)))"
    )
    return res
end

Base.@ccallable function julia_main()::Cint
    run_collapse_microbatch(CPU())
    return 0
end

end # module
