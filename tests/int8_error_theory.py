"""Check whether the int8 error matches int8-quantisation THEORY.

If the measured error equals the theoretical prediction, the kernel is behaving correctly
and my 1e-2 test bound was simply mis-set -- not a defect to "fix" by loosening a number.
Predicting from first principles is what makes the bound falsifiable rather than arbitrary.
"""
import math

print("=== int8 x-quantisation error: THEORY vs MEASURED (gate/up 640x2560, ntok=1) ===\n")

# Codebook kvalues_iq4nl (16 signed entries) -- from src/formats/weight/kvalues_iq4nl.inc.
kv = [-127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113]
kv_rms = math.sqrt(sum(v*v for v in kv)/len(kv))
print(f"  |kv| RMS               = {kv_rms:.2f}")

# Test generator: scales ~ U(-0.02, 0.02), so |d| RMS = 0.02/sqrt(3).
d_rms = 0.02 / math.sqrt(3)
print(f"  |d| RMS (U(-.02,.02))  = {d_rms:.5f}")

w_rms = kv_rms * d_rms
print(f"  => |w| RMS per element = {w_rms:.4f}")

# x ~ N(0,1) per element; quantised over a 32-block against that block's max|x|.
cols, nb = 2560, 2560//32
# E[max of 32 |N(0,1)|] ~= 2.2 sigma (empirical for n=32).
maxabs = 2.2
step = maxabs / 127.0                 # quantisation step
err_x_rms = step / math.sqrt(12)      # uniform in +-step/2 -> RMS step/sqrt(12)
print(f"\n  per-32-block max|x|    ~= {maxabs:.2f}")
print(f"  quant step             = {step:.5f}")
print(f"  x error RMS per elem   = {err_x_rms:.5f}")

# Per-term error = w * err_x ; summed over 2560 independent terms -> sqrt(2560) growth.
err_term = w_rms * err_x_rms
err_dot_rms = math.sqrt(cols) * err_term
print(f"\n  per-term error RMS     = {err_term:.5f}")
print(f"  dot error RMS (sqrt-N) = {err_dot_rms:.4f}")

# Measured (test output line 10): maxabs=6.660e-01, refRMS=40.9134, n=640.
meas_maxabs, meas_refRMS, n = 0.660, 40.9134, 640
# max over n=640 roughly-normal errors ~= 3.4 sigma.
sigma_max = 3.4
pred_maxabs = err_dot_rms * sigma_max
print(f"\n  PREDICTED maxabs (3.4sd of {n}) = {pred_maxabs:.4f}")
print(f"  MEASURED  maxabs              = {meas_maxabs:.4f}")
print(f"  ratio measured/predicted      = {meas_maxabs/pred_maxabs:.2f}x")

pred_rms_rel = err_dot_rms / meas_refRMS
meas_rms_rel = meas_maxabs / meas_refRMS
print(f"\n  PREDICTED error_rms/refRMS    = {pred_rms_rel:.3e}")
print(f"  MEASURED maxabs/refRMS        = {meas_rms_rel:.3e}  (max-based, ~3.4x the RMS figure)")
print(f"  => MEASURED error RMS / refRMS = {meas_rms_rel/sigma_max:.3e}")

print("\n=== VERDICT ===")
if abs(meas_maxabs/pred_maxabs - 1.0) < 0.25:
    print("  int8 error MATCHES quantisation theory within 25%.")
    print("  => The kernel is CORRECT; my 1e-2 bound was mis-set, not the code.")
    print(f"  Correct bound: maxabs/refRMS <= {meas_rms_rel*2:.1e} (2x measured, theory says")
    print(f"  ~{pred_rms_rel*3.4:.1e}); or gate on RMS-based error <= {pred_rms_rel*2:.1e}.")
else:
    print("  int8 error does NOT match theory -- real bug, investigate.")

print("\n=== Why the fp32 path shows 1-2 bf16 mismatches per few thousand (also expected) ===")
# fp32 accumulation error vs bf16 tie-point spacing.
fp32_maxabs = 6.485e-05           # measured, gate/up ntok=3
bf16_spacing = meas_refRMS * 2**-8  # bf16 has 8 mantissa bits
p_tie = 2 * fp32_maxabs / bf16_spacing
print(f"  fp32 maxabs            = {fp32_maxabs:.3e}")
print(f"  bf16 spacing @ |y|={meas_refRMS:.1f} = {bf16_spacing:.4f}")
print(f"  P(a value straddles a bf16 tie) ~= {p_tie:.2e}")
print(f"  expected mismatches in n=2560   ~= {p_tie*2560:.2f}")
print(f"  measured                        = 2   <-- matches")
print("  => 1-2 mismatches per few thousand is inherent fp32 tie-straddling, NOT a defect.")
print("     The GPU path has exactly the same property. An all-or-nothing bf16 gate is wrong;")
print("     gate on relative error (measured 7e-7..1.6e-6, well under Claude's 1e-5) instead.")
print("THEORY_DONE")
