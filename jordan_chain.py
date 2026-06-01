"""
jordan_chain.py
===============
Reproduces Tables C.1 and C.2 of Appendix C.

Appendix C.1 parameters:
  T_L = 0.5,  T_R = 3.2
  N   = 12000 quadrature points
  rtol=1e-13, atol=1e-15, DOP853

Algorithm (Appendix C.2):
  e1  : decaying mode, integrate T_R → T_L with WKB IC
  vm  : growing  mode, integrate T_R → T_L with WKB IC
  e2, e3 : variation-of-parameters + outgoing BC projection

Model operator:  P_mod = -∂²_x + x³  (h=1, energy 0)
Stokes contour:  x = t e^{i2π/5},  t ∈ ℝ_{>0}
After substitution:  v'' = t³ v,  outgoing ↔ v ~ t^{-3/4} exp(-(2/5)t^{5/2})

Matrix elements (bilinear, no conjugation):
  X_{jk} = e^{i4π/5} ∫_{T_L}^{T} t e_j(t) e_k(t) dt
  c_X    = X_{21} + X_{32}

Key result:  |X_{31}| / |c_X|  →  0  super-exponentially as T → ∞
             (WKB bound: log(|X31|/|cX|) ≤ -4T^{5/2}/5 + C)
"""

import numpy as np
from scipy.integrate import solve_ivp, simpson, cumulative_trapezoid
from scipy.interpolate import CubicSpline

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
T_L    = 0.5
T_R    = 3.2
N      = 12_000
PHASE2 = np.exp(1j * 4 * np.pi / 5)   # e^{i4π/5}
RTOL   = 1e-13
ATOL   = 1e-15

# ─────────────────────────────────────────────────────────────────────────────
# WKB INITIAL CONDITIONS  (Appendix C.1)
#
#   v±(T) = T^{-3/4} exp(±(2/5) T^{5/2})
#   v±'(T) = v±(T) · (-3/(4T) ± T^{3/2})
#
#   sign = +1 → decaying mode e1  (outgoing)
#   sign = -1 → growing  mode vm
# ─────────────────────────────────────────────────────────────────────────────
def wkb_ic(T, sign):
    v  = T**(-0.75) * np.exp(sign * (2.0 / 5.0) * T**2.5)
    dv = v * (-0.75 / T + sign * T**1.5)
    return [float(v), float(dv)]

def ode_hom(t, y):
    """Homogeneous ODE:  v'' = t³ v."""
    return [y[1], t**3 * y[0]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — integrate e1 and vm backward from T_R to T_L
# ─────────────────────────────────────────────────────────────────────────────
def integrate_basis(T_right, T_left, Npts):
    """
    Returns CubicSpline objects (e1_cs, e1d_cs, vm_cs, vmd_cs, W)
    on the interval [T_left, T_right].
    """
    t_bwd = np.linspace(T_right, T_left, Npts)
    t_asc = t_bwd[::-1]                          # ascending, for splines

    sol_e1 = solve_ivp(ode_hom, [T_right, T_left], wkb_ic(T_right, +1),
                       t_eval=t_bwd, method='DOP853', rtol=RTOL, atol=ATOL)
    sol_vm = solve_ivp(ode_hom, [T_right, T_left], wkb_ic(T_right, -1),
                       t_eval=t_bwd, method='DOP853', rtol=RTOL, atol=ATOL)

    e1_cs  = CubicSpline(t_asc, sol_e1.y[0][::-1])
    e1d_cs = CubicSpline(t_asc, sol_e1.y[1][::-1])
    vm_cs  = CubicSpline(t_asc, sol_vm.y[0][::-1])
    vmd_cs = CubicSpline(t_asc, sol_vm.y[1][::-1])

    # Wronskian W(e1, vm) — constant by Abel's identity
    W = e1_cs(T_right) * vmd_cs(T_right) - e1d_cs(T_right) * vm_cs(T_right)
    return e1_cs, e1d_cs, vm_cs, vmd_cs, W


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2/3 — build e_{k+1} by variation of parameters  (Appendix C.2)
#
# Particular solution:
#   u_part(t) = -vm(t) ∫[T_L→t] [e1(s)·src(s) / W] ds
#             + e1(t) ∫[T_L→t] [vm(s)·src(s) / W] ds
#
# Source:  src(t) = e^{i4π/5} · t · e_{k}(t)
#
# Outgoing BC (W(e_{k+1}, e1)|_{T_R} = 0):
#   C_{k+1} = -W(u_part, e1)|_{T_R} / W(vm, e1)|_{T_R}
#   e_{k+1} = u_part + C_{k+1} · vm
# ─────────────────────────────────────────────────────────────────────────────
def build_next(prev_cs, t_fwd,
               e1_cs, e1d_cs, vm_cs, vmd_cs, W, T_right,
               label="", verbose=True):
    e1g  = e1_cs(t_fwd);  e1dg = e1d_cs(t_fwd)
    vmg  = vm_cs(t_fwd);  vmdg = vmd_cs(t_fwd)
    src  = PHASE2 * t_fwd * prev_cs(t_fwd)

    int_f = cumulative_trapezoid(e1g * src / W, t_fwd, initial=0.0)
    int_g = cumulative_trapezoid(vmg * src / W, t_fwd, initial=0.0)

    u_part   = -vmg  * int_f + e1g  * int_g
    u_part_d = -vmdg * int_f + e1dg * int_g

    # Outgoing boundary condition at T_right
    W_u_e1  = u_part[-1]  * e1d_cs(T_right) - u_part_d[-1] * e1_cs(T_right)
    W_vm_e1 = vm_cs(T_right) * e1d_cs(T_right) - vmd_cs(T_right) * e1_cs(T_right)
    C_k = -W_u_e1 / W_vm_e1

    e_k_vals = u_part + C_k * vmg
    ek_cs = CubicSpline(t_fwd, e_k_vals)

    if verbose:
        # Residual check:  |P·e_k - src_prev| / |e_k|
        T_left = t_fwd[0]
        t_chk = np.linspace(T_left + 0.05, T_right - 0.05, 500)
        res = (ek_cs(t_chk, 2)
               - t_chk**3 * ek_cs(t_chk)
               + PHASE2 * t_chk * prev_cs(t_chk))
        rel = np.max(np.abs(res)) / (np.max(np.abs(ek_cs(t_chk))) + 1e-30)
        print(f"  {label}")
        print(f"    C_k              = {C_k:.4e}")
        print(f"    residual/|e_k|   = {rel:.2e}   (target < 1e-5)")

    return ek_cs


# ─────────────────────────────────────────────────────────────────────────────
# MATRIX ELEMENTS  (Appendix C.3)
# ─────────────────────────────────────────────────────────────────────────────
def matrix_elements(e1v, e2v, e3v, t):
    def Xjk(ej, ek):
        return PHASE2 * simpson(t * ej * ek, x=t)
    X21 = Xjk(e2v, e1v)
    X32 = Xjk(e3v, e2v)
    X31 = Xjk(e3v, e1v)
    cX  = X21 + X32
    return X21, X32, X31, cX


# ─────────────────────────────────────────────────────────────────────────────
# TABLE C.1 — WKB suppression ratio at multiple T values
# ─────────────────────────────────────────────────────────────────────────────
PAPER_TABLE_C1 = {3.2: 1.982e-3, 3.5: 8.511e-5, 4.0: 1.218e-7}
NQUAD_TABLE_C1 = {3.2: 7_000,    3.5: 8_000,   4.0: 10_000}


def run_table_c1():
    print("\n" + "=" * 62)
    print("TABLE C.1 — WKB suppression ratio  |X31| / |cX|")
    print("=" * 62)
    print(f"  {'T':>5}  {'|cX|':>12}  {'Code':>12}  {'Paper':>10}  {'Ratio':>6}  {'N':>7}")
    print(f"  {'-'*64}")

    for T_test in [3.2, 3.5, 4.0]:
        Nq = NQUAD_TABLE_C1[T_test]
        tf = np.linspace(T_L, T_test, Nq)

        e1c, e1dc, vmc, vmdc, Wt = integrate_basis(T_test, T_L, Nq)
        e2c = build_next(e1c,  tf, e1c, e1dc, vmc, vmdc, Wt, T_test, verbose=False)
        e3c = build_next(e2c,  tf, e1c, e1dc, vmc, vmdc, Wt, T_test, verbose=False)

        X21, X32, X31, cXt = matrix_elements(e1c(tf), e2c(tf), e3c(tf), tf)
        ratio = abs(X31) / abs(cXt)
        pr    = PAPER_TABLE_C1[T_test]
        print(f"  {T_test:>5.1f}  {abs(cXt):>12.4e}  {ratio:>12.3e}  {pr:>10.2e}  {ratio/pr:>6.2f}x  {Nq:>7}")
    print(f"  Note: |cX| is WKB-normalized; the ratio |X31|/|cX| is normalization-independent")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE C.2 — Jordan chain L²-norms at T = 3.2, N = 12000
# ─────────────────────────────────────────────────────────────────────────────
# WKB normalization: v±(T_R) = T_R^{-3/4} exp(±2T_R^{5/2}/5), giving W = -2.
# This code uses WKB normalization throughout.
# The paper (Appendix C.4, Table C.2) also uses WKB normalization for these norms.
# Note: paper text Appendix C.3 separately reports ||e1|| = 0.4731 under the
# algebraic normalization (condition <e_{k-1}, e_j> = delta_{j,k-1}); that value
# is NOT comparable to the WKB-normalized norms below.
PAPER_TABLE_C2 = {'e1': 1793.0, 'e2': 905615.0, 'e3': 4.568e8}   # WKB normalization


def run_table_c2(e1_cs, e2_cs, e3_cs, t_fwd):
    print("\n" + "=" * 62)
    print("TABLE C.2 — Jordan chain L²-norms  (T=3.2, N=12000, WKB norm)")
    print("=" * 62)
    print(f"  {'':>4}  {'Code':>12}  {'Paper (WKB)':>12}  Note")
    print(f"  {'-'*52}")
    for name, cs, pn in [('e1', e1_cs, PAPER_TABLE_C2['e1']),
                          ('e2', e2_cs, PAPER_TABLE_C2['e2']),
                          ('e3', e3_cs, PAPER_TABLE_C2['e3'])]:
        nm   = np.sqrt(simpson(np.abs(cs(t_fwd))**2, x=t_fwd))
        rel  = abs(nm - pn) / pn
        note = f"diff {rel:.1e}" if rel > 1e-3 else "✓"
        print(f"  {name:>4}  {nm:>12.4f}  {pn:>12.4f}  {note}")

    e1n = np.sqrt(simpson(np.abs(e1_cs(t_fwd))**2, x=t_fwd))
    e2n = np.sqrt(simpson(np.abs(e2_cs(t_fwd))**2, x=t_fwd))
    print(f"\n  c = ||e2|| / ||e1|| = {e2n/e1n:.4f}   (paper: 505)")
    print()
    print("  NOTE: L²-norms are WKB-normalized (W = -2); paper Appendix C.3")
    print("  reports ||e1|| = 0.4731 under the algebraic normalization (different).")
    print("  The ratio |X31|/|c_X| (Table C.1) is normalization-independent.")


# ─────────────────────────────────────────────────────────────────────────────
# WRONSKIAN STABILITY — Abel's identity check
# ─────────────────────────────────────────────────────────────────────────────
_wronskian_cache = {}   # {T_test: (e1c, e1dc, vmc, vmdc, W)}

def _get_basis(T_test):
    """integrate_basis sonuçlarını önbellekte sakla (run_wronskian için)."""
    if T_test not in _wronskian_cache:
        _wronskian_cache[T_test] = integrate_basis(T_test, T_L, 12_000)
    return _wronskian_cache[T_test]


def run_wronskian():
    print("\n" + "=" * 62)
    print("WRONSKIAN STABILITY — Abel's identity  W(e1, vm) = const")
    print("=" * 62)
    print(f"  {'T':>5}  {'W at T_R':>12}  {'W at T_L':>12}  {'|ΔW|/|W|':>10}")
    print(f"  {'-'*52}")
    for T_test in [3.2, 3.5, 4.0]:
        e1c, e1dc, vmc, vmdc, Wt = _get_basis(T_test)
        W_L = e1c(T_L) * vmdc(T_L) - e1dc(T_L) * vmc(T_L)
        dev = abs(Wt - W_L) / abs(Wt)
        print(f"  {T_test:>5.1f}  {Wt:>12.6f}  {W_L:>12.6f}  {dev:>10.2e}")
    print()
    print("  This code:   W = -2.000  (WKB IC: v±(T) = T^{-3/4} exp(±2T^{5/2}/5))")
    print("  Paper text (Appendix C.1): W ≈ +0.847  — DIFFERENT normalization of v_m.")
    print("  Both are correct; sign/scale of W cancels in the Jordan chain construction.")
    print("  The ratio |X31|/|c_X| is normalization-independent (Remark C.4).")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("STEP 1 — Integrate e1 and vm  (T_R=3.2, N=12000)")
    print("=" * 62)

    t_fwd = np.linspace(T_L, T_R, N)

    e1_cs, e1d_cs, vm_cs, vmd_cs, W = integrate_basis(T_R, T_L, N)
    W_L = e1_cs(T_L) * vmd_cs(T_L) - e1d_cs(T_L) * vm_cs(T_L)

    print(f"  W(e1,vm) @ T_R  = {W:.6f}")
    print(f"  W(e1,vm) @ T_L  = {W_L:.6f}  (Abel: should be constant)")
    print(f"  |ΔW|/|W|        = {abs(W-W_L)/abs(W):.2e}")
    print(f"  e1(T_R) / WKB   = {e1_cs(T_R) / wkb_ic(T_R,+1)[0]:.8f}  (should be 1)")

    print("\n" + "=" * 62)
    print("STEP 2/3 — Jordan chain construction  (Appendix C.2)")
    print("=" * 62)
    e2_cs = build_next(e1_cs, t_fwd, e1_cs, e1d_cs, vm_cs, vmd_cs, W, T_R,
                       label="e2  [P·e2 = e^{i4π/5} t e1]")
    e3_cs = build_next(e2_cs, t_fwd, e1_cs, e1d_cs, vm_cs, vmd_cs, W, T_R,
                       label="e3  [P·e3 = e^{i4π/5} t e2]")

    print("\n" + "=" * 62)
    print("MATRIX ELEMENTS  (Appendix C.3, T=3.2, N=12000)")
    print("=" * 62)
    e1v = e1_cs(t_fwd); e2v = e2_cs(t_fwd); e3v = e3_cs(t_fwd)
    X21, X32, X31, cX = matrix_elements(e1v, e2v, e3v, t_fwd)
    print(f"  |X21| = {abs(X21):.4e}")
    print(f"  |X32| = {abs(X32):.4e}")
    print(f"  |cX|  = {abs(cX):.4e}   (= |X21 + X32|)")
    print(f"  |X31| = {abs(X31):.4e}")
    print(f"\n  |X31| / |cX| = {abs(X31)/abs(cX):.4e}")
    print(f"  Paper (T=3.2): 1.982e-3")

    run_table_c1()
    run_table_c2(e1_cs, e2_cs, e3_cs, t_fwd)
    run_wronskian()
