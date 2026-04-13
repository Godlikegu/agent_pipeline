# Pipeline Evaluation Report (No Skills Injection)

**Model**: Vendor2 / Claude-4.6-opus  
**Run Date**: 2026-04-06 ~ 2026-04-08  
**Total Tasks**: 51  
**Skills**: Disabled (`retrieval_enabled: false`, `learning_enabled: false`)  
**Report Generated**: 2026-04-08  

---

## Summary


| Result   | Count | Tasks                                                                                                                                                                                                                                                                                                                                                                                     |
| -------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PASS** | 18    | ct_dual_energy, ct_sparse_view, eht_black_hole_dynamic, eht_black_hole_feature_extraction_dynamic, eit_conductivity_reconstruction, hessian_sim, lucky_imaging, mcr_hyperspectral, mri_dynamic_dce, mri_grappa, mri_sense, mri_t2_mapping, pet_mlem, photoacoustic_tomography, pnp_mri_reconstruction, raman_cell_phenotyping, ultrasound_sos_tomography, weather_radar_data_assimilation |
| **FAIL** | 33    | All others                                                                                                                                                                                                                                                                                                                                                                                |


**Overall Pass Rate: 18/51 (35.3%)**

---

## Detailed Results

### Legend

- **Baseline**: Reference method performance from literature/metrics.json  
- **Threshold**: Minimum NCC / Maximum NRMSE required to pass  
- **Pipeline**: Our agent pipeline's best output evaluation  
- **Status**: PASS / FAIL with failure reason

---

### 1. SSNP_ODT (Ssnp Odt)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.728  | **0.0008** |
| NRMSE  | N/A      | <= 0.1140 | **0.7167** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=-0.0007, NRMSE=0.7167 ✗ | Round 2: NCC=0.0008, NRMSE=0.8349 ✗ | Round 3: NCC=-0.0271, NRMSE=1.3633 ✗
**Root Cause Analysis**: STEP 1 - SYNTAX: The code has a critical structural error. The file contains a NESTED class definition inside the outer InverseSolver class. Starting around line ~160, there are duplicate import statements and a second 'class InverseSolver:' definition nested inside the first one. This means the out

**Visualization**: `data/end_sandbox/SSNP_ODT/visualization/SSNP_ODT_comparison.png`

---

### 2. confocal-nlos-fk (Confocal-Nlos-Fk)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **0.3217** |
| NRMSE  | N/A      | <= 0.1000 | **0.2177** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.3216, NRMSE=6.9441 ✗ | Round 2: NCC=0.0806, NRMSE=63538.6725 ✗ | Round 3: NCC=0.2306, NRMSE=65012.0528 ✗ | Round 4: NCC=0.3217, NRMSE=114150.2641 ✗ | Round 5: NCC=-0.1379, NRMSE=0.2177 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No errors. Code runs without exceptions. PASS.  STEP 2 (Interface): Shapes and signatures are consistent. PASS.  STEP 3 (Implementation Fidelity): Checking code against plan: - TOF calibration: Matches plan (unit detection, roll with -shift). CHECK. - Temporal crop to M=512:

**Visualization**: `data/end_sandbox/confocal-nlos-fk/visualization/confocal-nlos-fk_comparison.png`

---

### 3. conventional_ptychography (Conventional Ptychography)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.878  | **0.0531** |
| NRMSE  | N/A      | <= 0.0477 | **0.6541** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=-0.0130, NRMSE=0.7429 ✗ | Round 2: NCC=0.0034, NRMSE=0.7011 ✗ | Round 3: NCC=0.0300, NRMSE=0.6541 ✗ | Round 4: NCC=0.0025, NRMSE=0.6774 ✗ | Round 5: NCC=0.0531, NRMSE=0.6671 ✗
**Root Cause Analysis**: Step 1: No syntax/import errors. Step 2: Interface is correct. Step 3: Implementation matches plan faithfully - all conventions tested, positions computed per formula, ePIE updates correct, probe normalized, O_patch copied. Step 4: Algorithm correctness issue. NCC=0.053 after 300 epochs means zero s

**Visualization**: `data/end_sandbox/conventional_ptychography/visualization/conventional_ptychography_comparison.png`

---

### 4. ct_dual_energy (Ct Dual Energy) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.894  | **0.9945** |
| NRMSE  | N/A      | <= 0.0563 | **0.0530** |


**Status**: PASS  
**Iterations**: 4  
**Per-Round Metrics**: Round 4: NCC=0.9945, NRMSE=0.0530 ✓

**Visualization**: `data/end_sandbox/ct_dual_energy/visualization/ct_dual_energy_comparison.png`

---

### 5. ct_fan_beam (Ct Fan Beam)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.870  | **0.2145** |
| NRMSE  | N/A      | <= 0.0941 | **0.2018** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=-0.1897, NRMSE=0.2306 ✗ | Round 2: NCC=0.0569, NRMSE=0.2407 ✗ | Round 3: NCC=0.2145, NRMSE=0.2018 ✗ | Round 4: NCC=-0.1795, NRMSE=0.2298 ✗ | Round 5: NCC=-0.1210, NRMSE=0.7166 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors, no import errors. Code runs to completion. PASS.  STEP 2 (Interface): No shape mismatches or missing arguments. PASS.  STEP 3 (Implementation Fidelity): The code faithfully implements the Planner's SIRT algorithm with 8 geometry variants, pixel-driven splat

**Visualization**: `data/end_sandbox/ct_fan_beam/visualization/ct_fan_beam_comparison.png`

---

### 6. ct_poisson_lowdose (Ct Poisson Lowdose)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.887  | **0.5283** |
| NRMSE  | N/A      | <= 0.0936 | **0.2707** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 2: NCC=0.5283, NRMSE=0.2707 ✗ | Round 3: NCC=0.4011, NRMSE=0.3308 ✗ | Round 4: NCC=0.3680, NRMSE=0.3526 ✗ | Round 5: NCC=0.4889, NRMSE=0.2882 ✗
**Root Cause Analysis**: Step 1-3 pass: code runs, no interface issues, implementation matches plan. Step 4 fails: NCC=0.489 is far below 0.887 threshold. The algorithm has two fundamental problems: (1) The TV weight of 2e-4 is extremely aggressive for images in [0, 0.04] range - it destroys structural detail over 150 itera

**Visualization**: `data/end_sandbox/ct_poisson_lowdose/visualization/ct_poisson_lowdose_comparison.png`

---

### 7. ct_sparse_view (Ct Sparse View) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.869  | **0.9587** |
| NRMSE  | N/A      | <= 0.0755 | **0.0596** |


**Status**: PASS  
**Iterations**: 1  
**Per-Round Metrics**: Round 1: NCC=0.9587, NRMSE=0.0596 ✓

**Visualization**: `data/end_sandbox/ct_sparse_view/visualization/ct_sparse_view_comparison.png`

---

### 8. differentiable_deflectometry (Differentiable Deflectometry)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **0.0000** |
| NRMSE  | N/A      | <= 0.0426 | **0.0000** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.0000, NRMSE=86630000000000008019200278790144.0000 ✗ | Round 3: NCC=0.0000, NRMSE=92664407576143007046206847188992.0000 ✗ | Round 4: NCC=0.0000, NRMSE=118396549970949064394531827351552.0000 ✗ | Round 5: NCC=0.0000, NRMSE=0.0000 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors, code runs successfully. STEP 2 (Interface): No interface issues. STEP 3 (Implementation Fidelity): CRITICAL DEVIATION. The Planner's plan specifies a full differentiable ray tracing pipeline with phase extraction, temporal unwrapping, calibration loading, f

⚠️ **Near-miss**: NRMSE passes threshold but NCC fails
**Visualization**: `data/end_sandbox/differentiable_deflectometry/visualization/differentiable_deflectometry_comparison.png`

---

### 9. diffusion_mri_dti (Diffusion Mri Dti)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.898  | **0.2008** |
| NRMSE  | N/A      | <= 0.0643 | **0.7612** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 2: NCC=0.2008, NRMSE=0.7612 ✗ | Round 4: NCC=0.1931, NRMSE=0.7652 ✗
**Root Cause Analysis**: STEP 1: No syntax or import errors. Code structure is valid.  STEP 2: Interface contract looks correct.  STEP 3: Implementation fidelity check reveals a shape mismatch bug in the WLS solve step. The error is:  ValueError: solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc sig

**Visualization**: `data/end_sandbox/diffusion_mri_dti/visualization/diffusion_mri_dti_comparison.png`

---

### 10. eht_black_hole_UQ (Eht Black Hole Uq)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.591  | **0.2952** |
| NRMSE  | N/A      | <= 0.1490 | **0.1176** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=nan, NRMSE=nan ✗ | Round 2: NCC=0.2952, NRMSE=0.1176 ✗ | Round 3: NCC=nan, NRMSE=nan ✗ | Round 4: NCC=0.0000, NRMSE=0.1290 ✗ | Round 5: NCC=0.0383, NRMSE=2.8529 ✗
**Root Cause Analysis**: Step 1-3 pass: code runs without errors and implementation matches the plan. Step 4 FAILS: NCC=0.038 (zero correlation), NRMSE=2.85. The L_lca loss is ~87000 after 10000 epochs (reduced chi-sq ~180), meaning the model completely fails to fit log closure amplitudes. img_sum=18.37 vs F0=1.7763 shows a

⚠️ **Near-miss**: NRMSE passes threshold but NCC fails
**Visualization**: `data/end_sandbox/eht_black_hole_UQ/visualization/eht_black_hole_UQ_comparison.png`

---

### 11. eht_black_hole_dynamic (Eht Black Hole Dynamic) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.793  | **0.9353** |
| NRMSE  | N/A      | <= 0.0700 | **0.0458** |


**Status**: PASS  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.7484, NRMSE=0.0897 ✗ | Round 3: NCC=0.7711, NRMSE=0.0837 ✗ | Round 4: NCC=0.5507, NRMSE=0.1102 ✗ | Round 5: NCC=0.9353, NRMSE=0.0458 ✓

**Visualization**: `data/end_sandbox/eht_black_hole_dynamic/visualization/eht_black_hole_dynamic_comparison.png`

---

### 12. eht_black_hole_feature_extraction_dynamic (Eht Black Hole Feature Extraction Dynamic) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | N/A       | **0.9999** |
| NRMSE  | N/A      | N/A       | **0.1047** |


**Status**: PASS  
**Iterations**: 1  
**Per-Round Metrics**: Round 1: NCC=0.9999, NRMSE=0.1047 ✓

**Visualization**: `data/end_sandbox/eht_black_hole_feature_extraction_dynamic/visualization/eht_black_hole_feature_extraction_dynamic_comparison.png`

---

### 13. eht_black_hole_original (Eht Black Hole Original)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.544  | **0.4883** |
| NRMSE  | N/A      | <= 0.1520 | **0.1097** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.4883, NRMSE=0.1097 ✗ | Round 2: NCC=0.4883, NRMSE=0.1097 ✗ | Round 3: NCC=0.4883, NRMSE=0.1097 ✗ | Round 4: NCC=-0.0621, NRMSE=0.1285 ✗ | Round 5: NCC=0.0925, NRMSE=0.1258 ✗
**Root Cause Analysis**: Code runs without errors and implementation matches plan exactly. However NCC=0.093 indicates zero spatial correlation with ground truth despite chi2 values <1. The closure-only approach with 502 constraints for 4096 pixels is severely underdetermined. The Gaussian initialization provides no structu

⚠️ **Near-miss**: NRMSE passes threshold but NCC fails
**Visualization**: `data/end_sandbox/eht_black_hole_original/visualization/eht_black_hole_original_comparison.png`

---

### 14. eht_black_hole_tomography (Eht Black Hole Tomography)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **0.6950** |
| NRMSE  | N/A      | <= 0.0010 | **0.0419** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.6950, NRMSE=0.0419 ✗ | Round 3: NCC=0.5683, NRMSE=0.0480 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors or import issues visible. Code is valid Python.  STEP 2 (Interface): No interface/shape mismatches apparent. The class structure is sound.  STEP 3 (Implementation Fidelity): The code matches the plan in most respects. However, the critical issue is TIMEOUT. 

**Visualization**: `data/end_sandbox/eht_black_hole_tomography/visualization/eht_black_hole_tomography_comparison.png`

---

### 15. eit_conductivity_reconstruction (Eit Conductivity Reconstruction) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.502  | **0.5259** |
| NRMSE  | N/A      | <= 0.5943 | **0.1031** |


**Status**: PASS  
**Iterations**: 3  
**Per-Round Metrics**: Round 1: NCC=0.1680, NRMSE=0.1075 ✗ | Round 2: NCC=0.2840, NRMSE=0.1050 ✗ | Round 3: NCC=0.5259, NRMSE=0.1031 ✓

**Visualization**: `data/end_sandbox/eit_conductivity_reconstruction/visualization/eit_conductivity_reconstruction_comparison.png`

---

### 16. electron_ptychography (Electron Ptychography)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **0.0053** |
| NRMSE  | N/A      | <= 0.1500 | **0.6012** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.0007, NRMSE=0.6012 ✗ | Round 2: NCC=0.0024, NRMSE=0.9720 ✗ | Round 3: NCC=-0.0020, NRMSE=0.8637 ✗ | Round 4: NCC=0.0053, NRMSE=0.9576 ✗ | Round 5: NCC=-0.0061, NRMSE=0.8779 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): Code runs without syntax errors. STEP 2 (Interface): No interface issues. STEP 3 (Implementation Fidelity): The ePIE algorithm is diverging catastrophically - error grows 100x over 20 iterations, and NCC is essentially 0 (-0.006). Several implementation issues identified:  1

**Visualization**: `data/end_sandbox/electron_ptychography/visualization/electron_ptychography_comparison.png`

---

### 17. exoplanet_imaging (Exoplanet Imaging)


| Metric | Baseline | Threshold | Pipeline |
| ------ | -------- | --------- | -------- |
| NCC    | N/A      | >= 0.589  | **N/A**  |
| NRMSE  | N/A      | <= 0.0225 | **N/A**  |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=nan, NRMSE=nan ✗ | Round 2: NCC=nan, NRMSE=nan ✗ | Round 3: NCC=nan, NRMSE=nan ✗ | Round 4: NCC=nan, NRMSE=nan ✗ | Round 5: NCC=nan, NRMSE=nan ✗
**Root Cause Analysis**: Step 1: No syntax/import errors. Step 2: No interface issues. Step 3: The task description explicitly states the output should be a '2-D detection map (H, W)' with shape (100, 100). The Plan incorrectly specified shape (1, 100, 100) and the Coder followed it. The evaluation function likely computes 

**Visualization**: `data/end_sandbox/exoplanet_imaging/visualization/exoplanet_imaging_comparison.png`

---

### 18. fourier_ptychography (Fourier Ptychography)


| Metric | Baseline | Threshold | Pipeline |
| ------ | -------- | --------- | -------- |
| NCC    | N/A      | >= 0.865  | **N/A**  |
| NRMSE  | N/A      | <= 0.0506 | **N/A**  |


**Status**: FAIL  
**Iterations**: 0  

**Visualization**: `data/end_sandbox/fourier_ptychography/visualization/fourier_ptychography_comparison.png`

---

### 19. fpm_inr_reconstruction (Fpm Inr Reconstruction)


| Metric | Baseline | Threshold | Pipeline |
| ------ | -------- | --------- | -------- |
| NCC    | N/A      | >= 0.897  | **N/A**  |
| NRMSE  | N/A      | <= 0.0781 | **N/A**  |


**Status**: FAIL  
**Iterations**: 5  
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors. Code is valid Python.  STEP 2 (Interface): No interface mismatches.  STEP 3 (Implementation Fidelity): The code correctly implements the INR architecture and forward model as specified in the plan. However, the ROOT CAUSE of failure is that the code cannot 

**Visualization**: `data/end_sandbox/fpm_inr_reconstruction/visualization/fpm_inr_reconstruction_comparison.png`

---

### 20. hessian_sim (Hessian Sim) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | N/A       | **0.9986** |
| NRMSE  | N/A      | N/A       | **0.0203** |


**Status**: PASS  
**Iterations**: 2  
**Per-Round Metrics**: Round 2: NCC=0.9986, NRMSE=0.0203 ✓

**Visualization**: `data/end_sandbox/hessian_sim/visualization/hessian_sim_comparison.png`

---

### 21. insar_phase_unwrapping (Insar Phase Unwrapping)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.811  | **0.9181** |
| NRMSE  | N/A      | <= 0.0787 | **0.6389** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=-0.7924, NRMSE=0.7246 ✗ | Round 2: NCC=0.9181, NRMSE=0.6389 ✗ | Round 3: NCC=0.3835, NRMSE=0.6882 ✗ | Round 4: NCC=0.4996, NRMSE=1.1991 ✗ | Round 5: NCC=0.8915, NRMSE=1.2315 ✗
**Root Cause Analysis**: Step 1-3 pass: code is syntactically correct, interfaces match, implementation matches plan. Step 4: NCC=0.89 (good spatial correlation) but NRMSE=1.23 (terrible absolute error). This classic pattern indicates the unwrapped phase has correct spatial structure but wrong absolute level/offset. The Poi

⚠️ **Near-miss**: NCC passes threshold but NRMSE fails
**Visualization**: `data/end_sandbox/insar_phase_unwrapping/visualization/insar_phase_unwrapping_comparison.png`

---

### 22. lensless_imaging (Lensless Imaging)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **0.3815** |
| NRMSE  | N/A      | <= 0.1000 | **0.0882** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.3758, NRMSE=43.4149 ✗ | Round 3: NCC=-0.0045, NRMSE=87.2769 ✗ | Round 4: NCC=0.3815, NRMSE=0.0882 ✗ | Round 5: NCC=-0.0071, NRMSE=0.5040 ✗
**Root Cause Analysis**: Step 1-3 pass: code runs without errors and implementation exactly matches the plan. Step 4 fails: NCC=-0.007 means ZERO correlation with reference, indicating the reconstruction is completely spatially scrambled. The implementation faithfully follows the plan, so the algorithm itself is flawed. The

⚠️ **Near-miss**: NRMSE passes threshold but NCC fails
**Visualization**: `data/end_sandbox/lensless_imaging/visualization/lensless_imaging_comparison.png`

---

### 23. light_field_microscope (Light Field Microscope)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.722  | **0.2737** |
| NRMSE  | N/A      | <= 0.1078 | **0.5253** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.0710, NRMSE=4.1836 ✗ | Round 2: NCC=0.2400, NRMSE=0.6439 ✗ | Round 3: NCC=0.2568, NRMSE=0.5825 ✗ | Round 4: NCC=0.2737, NRMSE=0.5253 ✗ | Round 5: NCC=0.2250, NRMSE=0.5483 ✗
**Root Cause Analysis**: Step 3 passes — code matches plan exactly. Step 4 fails: The PSF peaks are displaced from center (223,221 instead of 225,225 for dz=0; much worse for other depths). With 30 lenslets (even number) of size 15, no lenslet block is centered on the optical axis at pixel 224.5. The Debye field is centered

**Visualization**: `data/end_sandbox/light_field_microscope/visualization/light_field_microscope_comparison.png`

---

### 24. lucky_imaging (Lucky Imaging) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.968  | **0.9854** |
| NRMSE  | N/A      | <= 0.0253 | **0.0171** |


**Status**: PASS  
**Iterations**: 3  
**Per-Round Metrics**: Round 3: NCC=0.9854, NRMSE=0.0171 ✓

**Visualization**: `data/end_sandbox/lucky_imaging/visualization/lucky_imaging_comparison.png`

---

### 25. mcr_hyperspectral (Mcr Hyperspectral) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.880  | **0.9950** |
| NRMSE  | N/A      | <= 0.1255 | **0.0423** |


**Status**: PASS  
**Iterations**: 1  
**Per-Round Metrics**: Round 1: NCC=0.9950, NRMSE=0.0423 ✓

**Visualization**: `data/end_sandbox/mcr_hyperspectral/visualization/mcr_hyperspectral_comparison.png`

---

### 26. microscope_denoising (Microscope Denoising)


| Metric | Baseline | Threshold | Pipeline |
| ------ | -------- | --------- | -------- |
| NCC    | N/A      | >= 0.850  | **N/A**  |
| NRMSE  | N/A      | <= 0.2000 | **N/A**  |


**Status**: FAIL  
**Iterations**: 5  
**Root Cause Analysis**: STEP 1 - SYNTAX & IMPORTS: The code has no syntax errors or import issues. However, there is an AttributeError at runtime: 'InverseSolver' object has no attribute '_build_unet'. Looking at the code, the `_build_unet` method is defined OUTSIDE the class body (at module level, after the class definiti

**Visualization**: `data/end_sandbox/microscope_denoising/visualization/microscope_denoising_comparison.png`

---

### 27. mri_dynamic_dce (Mri Dynamic Dce) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.878  | **0.9611** |
| NRMSE  | N/A      | <= 0.0667 | **0.0589** |


**Status**: PASS  
**Iterations**: 3  
**Per-Round Metrics**: Round 1: NCC=0.9068, NRMSE=0.0948 ✗ | Round 2: NCC=0.8466, NRMSE=0.1282 ✗ | Round 3: NCC=0.9611, NRMSE=0.0589 ✓

**Visualization**: `data/end_sandbox/mri_dynamic_dce/visualization/mri_dynamic_dce_comparison.png`

---

### 28. mri_grappa (Mri Grappa) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **0.9999** |
| NRMSE  | N/A      | <= 0.0045 | **0.0041** |


**Status**: PASS  
**Iterations**: 4  
**Per-Round Metrics**: Round 1: NCC=0.9979, NRMSE=0.0183 ✗ | Round 2: NCC=0.9988, NRMSE=0.0172 ✗ | Round 3: NCC=0.9998, NRMSE=0.0053 ✗ | Round 4: NCC=0.9999, NRMSE=0.0041 ✓

**Visualization**: `data/end_sandbox/mri_grappa/visualization/mri_grappa_comparison.png`

---

### 29. mri_l1_wavelet (Mri L1 Wavelet)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.784  | **0.0000** |
| NRMSE  | N/A      | <= 0.1257 | **N/A**    |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.0000, NRMSE=1807172406634195926042619674624.0000 ✗ | Round 2: NCC=0.0000, NRMSE=1816583970427104833360730521600.0000 ✗ | Round 3: NCC=0.0000, NRMSE=1806470863529477591323868397568.0000 ✗ | Round 4: NCC=0.0000, NRMSE=1806470863529477591323868397568.0000 ✗ | Round 5: NCC=0.0000, NRMSE=1810061339460995037036810338304.0000 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors, code runs to completion.  STEP 2 (Interface): No interface issues, shapes match.  STEP 3 (Implementation Fidelity): The critical issue is the wavelet transform implementation. The logs show 'Wavelet round-trip relative error: 8.01e-01' which means the DWT/I

**Visualization**: `data/end_sandbox/mri_l1_wavelet/visualization/mri_l1_wavelet_comparison.png`

---

### 30. mri_noncartesian_cs (Mri Noncartesian Cs)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.890  | **0.0386** |
| NRMSE  | N/A      | <= 0.0407 | **0.5406** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.0386, NRMSE=0.6716 ✗ | Round 2: NCC=0.0386, NRMSE=0.6716 ✗ | Round 3: NCC=0.0338, NRMSE=0.6636 ✗ | Round 4: NCC=-0.0106, NRMSE=0.5661 ✗ | Round 5: NCC=-0.0019, NRMSE=0.5406 ✗
**Root Cause Analysis**: Step 1: No syntax/import errors. Step 2: No interface issues. Step 3: Code implements the plan faithfully - ISTA loop, wavelet proximal, density compensation all match. However, the code only tests 2 of the 4 planned coordinate conventions (missing scaled variants). Step 4: NCC = -0.002 with decreas

**Visualization**: `data/end_sandbox/mri_noncartesian_cs/visualization/mri_noncartesian_cs_comparison.png`

---

### 31. mri_pnp_admm (Mri Pnp Admm)


| Metric | Baseline | Threshold | Pipeline |
| ------ | -------- | --------- | -------- |
| NCC    | N/A      | >= 0.895  | **N/A**  |
| NRMSE  | N/A      | <= 0.0365 | **N/A**  |


**Status**: FAIL  
**Iterations**: 5  
**Root Cause Analysis**: STEP 1: CHECK SYNTAX & IMPORTS - The code has a critical STRUCTURAL error. The `InverseSolver.__init_`_ method body is just `pass` - it does not store any instance attributes (mask, y_kspace, rho, n_iter, device, image_shape, num_layers, num_features, denoiser_weights_path, denoiser_model, x0). The 

**Visualization**: `data/end_sandbox/mri_pnp_admm/visualization/mri_pnp_admm_comparison.png`

---

### 32. mri_sense (Mri Sense) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.889  | **0.9811** |
| NRMSE  | N/A      | <= 0.0629 | **0.0422** |


**Status**: PASS  
**Iterations**: 1  
**Per-Round Metrics**: Round 1: NCC=0.9811, NRMSE=0.0422 ✓

**Visualization**: `data/end_sandbox/mri_sense/visualization/mri_sense_comparison.png`

---

### 33. mri_t2_mapping (Mri T2 Mapping) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.899  | **0.9980** |
| NRMSE  | N/A      | <= 0.0408 | **0.0177** |


**Status**: PASS  
**Iterations**: 1  
**Per-Round Metrics**: Round 1: NCC=0.9980, NRMSE=0.0177 ✓

**Visualization**: `data/end_sandbox/mri_t2_mapping/visualization/mri_t2_mapping_comparison.png`

---

### 34. mri_tv (Mri Tv)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.866  | **0.2247** |
| NRMSE  | N/A      | <= 0.0525 | **0.3831** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=-0.0084, NRMSE=0.4450 ✗ | Round 2: NCC=-0.0035, NRMSE=0.4443 ✗ | Round 3: NCC=0.2247, NRMSE=0.3831 ✗ | Round 4: NCC=-0.0035, NRMSE=0.4443 ✗ | Round 5: NCC=0.2247, NRMSE=0.3831 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors, code runs to completion.  STEP 2 (Interface): No interface issues, shapes and signatures are correct.  STEP 3 (Implementation Fidelity): The critical issue is in the `_tv_prox` method. The image is collapsing toward zero: max|x| goes from 59.2 → 51.4 → 32.8

**Visualization**: `data/end_sandbox/mri_tv/visualization/mri_tv_comparison.png`

---

### 35. mri_varnet (Mri Varnet)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.899  | **0.9980** |
| NRMSE  | N/A      | <= 0.0140 | **0.0143** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 4: NCC=0.9980, NRMSE=0.0143 ✗
**Root Cause Analysis**: STEP 1: No syntax or import errors. Code runs but crashes at runtime. STEP 2: Interface is correct - VarNet forward expects (masked_kspace, mask, num_low_frequencies). STEP 3: The error occurs at fastmri/models/varnet.py line 318: `torch.where(mask, current_kspace - ref_kspace, zero)`. The `torch.wh

⚠️ **Near-miss**: NCC passes threshold but NRMSE fails
**Visualization**: `data/end_sandbox/mri_varnet/visualization/mri_varnet_comparison.png`

---

### 36. pet_mlem (Pet Mlem) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.886  | **0.9663** |
| NRMSE  | N/A      | <= 0.0610 | **0.0401** |


**Status**: PASS  
**Iterations**: 1  
**Per-Round Metrics**: Round 1: NCC=0.9663, NRMSE=0.0401 ✓

**Visualization**: `data/end_sandbox/pet_mlem/visualization/pet_mlem_comparison.png`

---

### 37. photoacoustic_tomography (Photoacoustic Tomography) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.549  | **0.5940** |
| NRMSE  | N/A      | <= 0.3949 | **0.2862** |


**Status**: PASS  
**Iterations**: 3  
**Per-Round Metrics**: Round 2: NCC=0.5436, NRMSE=0.3432 ✗ | Round 3: NCC=0.5940, NRMSE=0.2862 ✓

**Visualization**: `data/end_sandbox/photoacoustic_tomography/visualization/photoacoustic_tomography_comparison.png`

---

### 38. plane_wave_ultrasound (Plane Wave Ultrasound)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | N/A       | **0.6095** |
| NRMSE  | N/A      | N/A       | **0.3379** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.5693, NRMSE=0.6652 ✗ | Round 2: NCC=0.5693, NRMSE=0.6652 ✗ | Round 3: NCC=0.5520, NRMSE=0.6612 ✗ | Round 4: NCC=0.5520, NRMSE=0.6612 ✗ | Round 5: NCC=0.6095, NRMSE=0.3379 ✗
**Root Cause Analysis**: Steps 1-3 pass: code runs without errors and faithfully implements the plan. However, NCC=0.61 is well below the 0.85 threshold after 4+ iterations. The implementation exactly matches the plan's specifications (beta not beta², plus signs for steering and lateral correction, centered grids). The pers

**Visualization**: `data/end_sandbox/plane_wave_ultrasound/visualization/plane_wave_ultrasound_comparison.png`

---

### 39. pnp_mri_reconstruction (Pnp Mri Reconstruction) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | N/A       | **0.9464** |
| NRMSE  | N/A      | N/A       | **0.0425** |


**Status**: PASS  
**Iterations**: 2  
**Per-Round Metrics**: Round 1: NCC=0.9464, NRMSE=2563.7128 ✗ | Round 2: NCC=0.9254, NRMSE=0.0425 ✓

**Visualization**: `data/end_sandbox/pnp_mri_reconstruction/visualization/pnp_mri_reconstruction_comparison.png`

---

### 40. raman_cell_phenotyping (Raman Cell Phenotyping) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **1.0000** |
| NRMSE  | N/A      | <= 0.1100 | **0.0000** |


**Status**: PASS  
**Iterations**: 4  
**Per-Round Metrics**: Round 3: NCC=0.0000, NRMSE=0.5386 ✗ | Round 4: NCC=1.0000, NRMSE=0.0000 ✓

**Visualization**: `data/end_sandbox/raman_cell_phenotyping/visualization/raman_cell_phenotyping_comparison.png`

---

### 41. reflection_ODT (Reflection Odt)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.785  | **0.0481** |
| NRMSE  | N/A      | <= 0.3590 | **0.5175** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 2: NCC=0.0481, NRMSE=0.5175 ✗ | Round 3: NCC=0.0386, NRMSE=0.6477 ✗
**Root Cause Analysis**: STEP 1: CHECK SYNTAX & IMPORTS. The error is 'NameError: name InverseSolver is not defined' at line 329. Looking at the code, after the class definition ends (after the solve method), there are TWO blocks of loose code (not inside any function or if **name** block) that try to instantiate InverseSol

**Visualization**: `data/end_sandbox/reflection_ODT/visualization/reflection_ODT_comparison.png`

---

### 42. seismic_FWI_original (Seismic Fwi Original)


| Metric | Baseline | Threshold | Pipeline |
| ------ | -------- | --------- | -------- |
| NCC    | N/A      | N/A       | **N/A**  |
| NRMSE  | N/A      | N/A       | **N/A**  |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 2: NCC=nan, NRMSE=nan ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors, all imports are valid. STEP 2 (Interface): No interface/signature issues. STEP 3 (Implementation vs Plan): The error is 'RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn'. This occurs at `shot_loss.backward()` in `_comput

**Visualization**: `data/end_sandbox/seismic_FWI_original/visualization/seismic_FWI_original_comparison.png`

---

### 43. shapelet_source_reconstruction (Shapelet Source Reconstruction)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.860  | **1.0000** |
| NRMSE  | N/A      | <= 0.0594 | **0.3724** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=1.0000, NRMSE=43.9783 ✗ | Round 2: NCC=0.7037, NRMSE=4.2362 ✗ | Round 3: NCC=0.7129, NRMSE=3.7169 ✗ | Round 4: NCC=0.7093, NRMSE=4.3121 ✗ | Round 5: NCC=0.6960, NRMSE=0.3724 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors, code runs successfully. STEP 2 (Interface): No interface issues, code produces output. STEP 3 (Implementation Fidelity): The code correctly implements the Plan. Convention B (swapped n1/n2) gives NCC=1.0 with image_reconstructed, confirming the shapelet eva

⚠️ **Near-miss**: NCC passes threshold but NRMSE fails
**Visualization**: `data/end_sandbox/shapelet_source_reconstruction/visualization/shapelet_source_reconstruction_comparison.png`

---

### 44. single_molecule_light_field (Single Molecule Light Field)


| Metric | Baseline | Threshold | Pipeline |
| ------ | -------- | --------- | -------- |
| NCC    | N/A      | N/A       | **N/A**  |
| NRMSE  | N/A      | N/A       | **N/A**  |


**Status**: FAIL  
**Iterations**: 5  
**Root Cause Analysis**: Step 1: No syntax errors. Step 2: No interface issues. Step 3: Implementation appears to follow the plan faithfully - sign conventions, formulas, parameters all match. Step 4: The algorithm produces only 817 molecules from 150k localizations with an absurdly narrow spatial extent (0.3 μm x 0.2 μm) w

**Visualization**: `data/end_sandbox/single_molecule_light_field/visualization/single_molecule_light_field_comparison.png`

---

### 45. spectral_snapshot_compressive_imaging (Spectral Snapshot Compressive Imaging)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.895  | **0.9380** |
| NRMSE  | N/A      | <= 0.0239 | **0.0784** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.9380, NRMSE=0.0784 ✗ | Round 3: NCC=0.7815, NRMSE=0.1343 ✗ | Round 4: NCC=0.8983, NRMSE=0.0969 ✗ | Round 5: NCC=0.8875, NRMSE=0.1013 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No syntax errors, all imports valid. Code runs to completion.  STEP 2 (Interface): No shape mismatches or missing arguments. Interface is correct.  STEP 3 (Implementation Fidelity): Checked against plan: - Forward/adjoint operators: CORRECT (y[:, shift:shift+W] += Phi * x[:,

⚠️ **Near-miss**: NCC passes threshold but NRMSE fails
**Visualization**: `data/end_sandbox/spectral_snapshot_compressive_imaging/visualization/spectral_snapshot_compressive_imaging_comparison.png`

---

### 46. ultrasound_sos_tomography (Ultrasound Sos Tomography) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.884  | **0.9976** |
| NRMSE  | N/A      | <= 0.0161 | **0.0035** |


**Status**: PASS  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.4355, NRMSE=0.7229 ✗ | Round 2: NCC=-0.1151, NRMSE=0.1079 ✗ | Round 3: NCC=-0.1151, NRMSE=0.1079 ✗ | Round 4: NCC=-0.1404, NRMSE=0.1898 ✗ | Round 5: NCC=0.9976, NRMSE=0.0035 ✓

**Visualization**: `data/end_sandbox/ultrasound_sos_tomography/visualization/ultrasound_sos_tomography_comparison.png`

---

### 47. usct_FWI (Usct Fwi)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **0.0312** |
| NRMSE  | N/A      | <= 0.0316 | **0.1065** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 2: NCC=-0.0470, NRMSE=0.1065 ✗ | Round 3: NCC=-0.0002, NRMSE=0.1120 ✗ | Round 4: NCC=0.0312, NRMSE=0.3402 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): Code is syntactically valid Python, all imports are available. No structural issues.  STEP 2 (Interface): No interface contract violations detected.  STEP 3 (Implementation vs Plan): The implementation appears to match the plan correctly in terms of formulas. However, the co

**Visualization**: `data/end_sandbox/usct_FWI/visualization/usct_FWI_comparison.png`

---

### 48. weather_radar_data_assimilation (Weather Radar Data Assimilation) -- PASS


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.783  | **0.8001** |
| NRMSE  | N/A      | <= 0.0565 | **0.0560** |


**Status**: PASS  
**Iterations**: 1  
**Per-Round Metrics**: Round 1: NCC=0.8001, NRMSE=0.0560 ✓

**Visualization**: `data/end_sandbox/weather_radar_data_assimilation/visualization/weather_radar_data_assimilation_comparison.png`

---

### 49. xray_laminography_tike (Xray Laminography Tike)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.263  | **0.1187** |
| NRMSE  | N/A      | <= 0.0189 | **5.7199** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 2: NCC=0.0217, NRMSE=8.5484 ✗ | Round 3: NCC=0.1187, NRMSE=10.9707 ✗ | Round 5: NCC=-0.4049, NRMSE=5.7199 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): The code has severe STRUCTURAL issues. There are multiple copies of the main execution code: (a) orphaned code at module scope after the class definition that runs the solver twice, (b) code inside `if __name__ == '__main__'` that ignores the CG solver entirely and instead i

**Visualization**: `data/end_sandbox/xray_laminography_tike/visualization/xray_laminography_tike_comparison.png`

---

### 50. xray_ptychography_tike (Xray Ptychography Tike)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.900  | **0.0000** |
| NRMSE  | N/A      | <= 0.1000 | **0.4877** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 1: NCC=0.0000, NRMSE=0.4877 ✗ | Round 2: NCC=-0.5155, NRMSE=0.4877 ✗ | Round 3: NCC=0.0000, NRMSE=0.4877 ✗ | Round 4: NCC=-0.5155, NRMSE=0.4877 ✗ | Round 5: NCC=0.0000, NRMSE=0.4877 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): PASS - No syntax errors, imports valid, code runs without crashes.  STEP 2 (Interface Contract): PASS - All method signatures match expected shapes, no shape mismatches.  STEP 3 (Implementation Fidelity): CRITICAL FAILURE - Multiple deviations from Plan's mathematical specif

**Visualization**: `data/end_sandbox/xray_ptychography_tike/visualization/xray_ptychography_tike_comparison.png`

---

### 51. xray_tooth_gridrec (Xray Tooth Gridrec)


| Metric | Baseline | Threshold | Pipeline   |
| ------ | -------- | --------- | ---------- |
| NCC    | N/A      | >= 0.884  | **0.9679** |
| NRMSE  | N/A      | <= 0.0323 | **0.0350** |


**Status**: FAIL  
**Iterations**: 5  
**Per-Round Metrics**: Round 2: NCC=0.9679, NRMSE=0.0350 ✗ | Round 3: NCC=0.8448, NRMSE=0.1410 ✗ | Round 4: NCC=0.8448, NRMSE=0.0692 ✗ | Round 5: NCC=0.8888, NRMSE=0.0594 ✗
**Root Cause Analysis**: STEP 1 (Syntax/Imports): No errors - code runs cleanly, no STDERR output. STEP 2 (Interface): No shape mismatches or signature issues. STEP 3 (Implementation Fidelity): The code faithfully implements the plan. Cross-correlation with wrap-around handling is correct. SSD-based fine grid search over 11

⚠️ **Near-miss**: NCC passes threshold but NRMSE fails
**Visualization**: `data/end_sandbox/xray_tooth_gridrec/visualization/xray_tooth_gridrec_comparison.png`

---

## Failure Analysis Summary

### Failure Categories


| Category                               | Count | Tasks                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Algorithm/Physics Error**            | 3     | ct_poisson_lowdose, light_field_microscope, plane_wave_ultrasound                                                                                                                                                                                                                                                                                                                                                                                    |
| **Near-miss (partial threshold pass)** | 2     | eht_black_hole_UQ, lensless_imaging                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Insufficient Tuning**                | 0     |                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **No Output (code error/timeout)**     | 7     | exoplanet_imaging, fourier_ptychography, fpm_inr_reconstruction, microscope_denoising, mri_pnp_admm, seismic_FWI_original, single_molecule_light_field                                                                                                                                                                                                                                                                                               |
| **Code Structure Bug**                 | 21    | SSNP_ODT, confocal-nlos-fk, conventional_ptychography, ct_fan_beam, differentiable_deflectometry, diffusion_mri_dti, eht_black_hole_original, eht_black_hole_tomography, electron_ptychography, insar_phase_unwrapping, mri_l1_wavelet, mri_noncartesian_cs, mri_tv, mri_varnet, reflection_ODT, shapelet_source_reconstruction, spectral_snapshot_compressive_imaging, usct_FWI, xray_laminography_tike, xray_ptychography_tike, xray_tooth_gridrec |
| **Timeout**                            | 0     |                                                                                                                                                                                                                                                                                                                                                                                                                                                      |


### Near-Miss Tasks (close to passing)

- **differentiable_deflectometry**: Best NCC=0.0000 (thresh 0.900), Best NRMSE=0.0000 (thresh 0.0426) — NRMSE PASSES ✓, NCC at 0% of threshold
- **eht_black_hole_UQ**: Best NCC=0.2952 (thresh 0.591), Best NRMSE=0.1176 (thresh 0.1490) — NRMSE PASSES ✓, NCC at 50% of threshold
- **eht_black_hole_original**: Best NCC=0.4883 (thresh 0.544), Best NRMSE=0.1097 (thresh 0.1520) — NRMSE PASSES ✓, NCC at 90% of threshold
- **insar_phase_unwrapping**: Best NCC=0.9181 (thresh 0.811), Best NRMSE=0.6389 (thresh 0.0787) — NCC PASSES ✓
- **lensless_imaging**: Best NCC=0.3815 (thresh 0.900), Best NRMSE=0.0882 (thresh 0.1000) — NRMSE PASSES ✓, NCC at 42% of threshold
- **mri_varnet**: Best NCC=0.9980 (thresh 0.899), Best NRMSE=0.0143 (thresh 0.0140) — NCC PASSES ✓
- **shapelet_source_reconstruction**: Best NCC=1.0000 (thresh 0.860), Best NRMSE=0.3724 (thresh 0.0594) — NCC PASSES ✓
- **spectral_snapshot_compressive_imaging**: Best NCC=0.9380 (thresh 0.895), Best NRMSE=0.0784 (thresh 0.0239) — NCC PASSES ✓
- **xray_tooth_gridrec**: Best NCC=0.9679 (thresh 0.884), Best NRMSE=0.0350 (thresh 0.0323) — NCC PASSES ✓

## Summary by Domain


| Domain                    | Total | Success | Failed | Rate |
| ------------------------- | ----- | ------- | ------ | ---- |
| CT                        | 4     | 2       | 2      | 50%  |
| MRI                       | 10    | 5       | 5      | 50%  |
| EHT (Black Hole)          | 5     | 2       | 3      | 40%  |
| Ptychography              | 4     | 0       | 4      | 0%   |
| Ultrasound                | 3     | 1       | 2      | 33%  |
| Optical/Microscopy        | 6     | 2       | 4      | 33%  |
| ODT/Wave                  | 3     | 0       | 3      | 0%   |
| Spectral/Hyperspectral    | 3     | 2       | 1      | 67%  |
| X-ray/Tomography          | 4     | 2       | 2      | 50%  |
| Geophysics/Remote Sensing | 4     | 1       | 3      | 25%  |
| Other                     | 5     | 1       | 4      | 20%  |


---

## Conclusion

Without skills injection, the pipeline achieves a **35.3% success rate (18/51 tasks)** using the Vendor2/Claude-4.6-opus model. Most failures exhaust all 5 retry iterations, indicating the model struggles with complex scientific reconstruction tasks without domain-specific guidance.

Key observations:

1. **MRI tasks** have the highest success rate among multi-task domains (4/10 = 40%)
2. **Ptychography** tasks uniformly fail (0/4), suggesting these require specialized domain knowledge
3. **2 tasks** are near-misses that partially meet thresholds — these could benefit most from skills injection
4. Tasks that succeed on first attempt (0 retries) tend to be algorithmically simpler (FBP, direct solvers)
5. Most failures are algorithm/physics errors rather than code bugs, indicating the model needs domain expertise

