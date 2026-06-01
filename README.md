# sds-jordan-chain

Numerical ancillary files for:

> **Microlocal Spectral Bifurcation in Schwarzschild–de Sitter Geometry**

These scripts reproduce all numerical results in Appendix C (Jordan chain
verification), Appendix B.2 (Schwarzian contribution), and Section 6
(robustness tests).

## Repository contents

| File | Reproduces | Status |
|---|---|---|
| `jordan_chain.py` | Tables C.1, C.2 — WKB suppression ratios and L²-norms | ✓ |
| `schwarzian.py` | Appendix B.2 — critical point, S_x(r^c), Q^Schw_31 | ✓ |
| `inner_product_e3e1.py` | Appendix C.3 — ⟨e₃,e₁⟩_{L²(Γ)}, with finite-difference cross-check | ✓ |
| `robustness_tests.py` | Section 6 — genuine operator sensitivity and contour rotation | ✓ |
| `requirements.txt` | Python dependencies | |

## Quick start

```bash
pip install -r requirements.txt
python jordan_chain.py          # Tables C.1 and C.2
python schwarzian.py            # Appendix B.2
python inner_product_e3e1.py    # ⟨e₃, e₁⟩ cross-check with FD verification
python robustness_tests.py      # Section 6 robustness tests
```

## Parameters (Appendix C.1)

- Interval: t ∈ [T_L, T_R] = [0.5, 3.2]
- Quadrature points: N = 12 000 (Tables C.1–C.2)
- ODE solver: DOP853, rtol=1e-13, atol=1e-15
- WKB initial data at t = T_R
- ODE: v''(t) = t³·v(t)  [real coefficient, θ=2π/5 → e^{i5θ}=1]

## Expected output summary

**Table C.1 — WKB suppression ratio |X₃₁|/|c_X|** (primary result):
```
T     Code
3.2   1.982e-3
3.5   8.511e-5
4.0   1.218e-7
```
Ratio invariant under normalization (Remark C.4). ✓

**Table C.2 — L²-norms** (WKB normalization):
```
‖e₁‖ = 1793
‖e₂‖ = 9.056×10⁵
‖e₃‖ = 4.568×10⁸
c = ‖e₂‖/‖e₁‖ = 505
```

**Appendix B.2 — Schwarzian:**
```
r^c ≈ 0.9109,  m_c = r_s/2 ≈ 0.5596,  S_x(r^c) ≈ -0.6617  ✓
|Q^Schw_31| / |c_X| ≈ 8×10⁻⁴  (subleading ✓)
```

**Section 6 — Robustness:**
```
x³ vs x²: |X31/cX| for x³ → 0 super-exp; x² decays much more slowly ✓
           (each operator gets its own independently built Jordan chain)
δ=0.05 rotation: ratio ≈ 9.85×10⁻⁵ at T=3.5, ≈ 1.49×10⁻⁷ at T=4.0 (robust ✓)
```

## Notes on normalization (Remark C.4)

The ratio |X₃₁|/|c_X| is invariant under rescaling eⱼ → α eⱼ.
The L²-norms in Table C.2 depend on the normalization of v_m.
This code uses the WKB normalization v_±(T_R) = T_R^{-3/4} exp(±2T_R^{5/2}/5),
which gives Wronskian W = -2 and ‖e₁‖ = 1793.
The paper (v39) reports these same WKB-normalized values.
The ratio |X₃₁|/|c_X| is unaffected by normalization choice.

## Fixes applied in this version

| File | Issue | Fix |
|---|---|---|
| `schwarzian.py` | `scipy.misc.derivative` removed in scipy 2.0 | Replaced with `sympy` analytic derivatives |
| `schwarzian.py` | `fsolve` converged to wrong root | Grid search; added residual guard; `except: pass` → `except Exception` |
| `schwarzian.py` | Ambiguous sympy variable names (`r_s` vs `rs_sym`) | Renamed to `rv_sym`, `rs_sym`; added comment on derivative direction |
| `schwarzian.py` | Normalization of Q31/cX ratio unexplained | Added inline comment proving normalization cancels |
| `robustness_tests.py` | Contour rotation didn't update ODE coefficient | Full rotation: `v'' = t³·e^{i5θ}·v`, WKB IC updated |
| `robustness_tests.py` | Test 1 used same Jordan chain for x and x² | Now rebuilds chain independently for `P=-∂²+x²` (n=2, θ=π/3) |
| `robustness_tests.py` | Parity argument ("e₁(EVEN)×e₃(EVEN)→0") stated without basis | Removed; WKB analytic bound cited instead |
| `robustness_tests.py` | `build_chain` had wrong ODE coeff (n+1 vs n+2), source phase (nθ vs 2θ), WKB kappa ((n-2)/4 vs n/4), and reversed CubicSpline indexing | All four fixed; x³ values now match Table C.1 exactly (ratio 1.0000) |
| `inner_product_e3e1.py` | Was a copy of jordan_chain.py, not an independent check | Added finite-difference residual cross-check (Method B) |
| `inner_product_e3e1.py` | Phase factor difference PHASE1 vs PHASE2 unexplained | Added explicit comment: PHASE1 for ∫e₃e₁, PHASE2 for ∫t·ej·ek |
| `jordan_chain.py` | `run_wronskian` recomputed integrations for each T | Added `_wronskian_cache`; removed unused `t_bwd` variable |

## Inner products

Two distinct pairings appear (Appendix C.1):
- **Sesquilinear** ⟨u,v⟩ = ∫ u·v̄ (with conjugation): used in Grushin framework
- **Bilinear** ⟨u,v⟩_Γ = e^{i2π/5}∫ u·v (no conjugation, no t weight): used for ⟨e₃,e₁⟩
- **Matrix elements** X_{jk} = e^{i4π/5}∫ t·ej·ek dt (bilinear with t weight)

The difference between PHASE1 (e^{i2π/5}) and PHASE2 (e^{i4π/5}) reflects
the presence or absence of the t weight factor, as defined in Appendix C.1.
