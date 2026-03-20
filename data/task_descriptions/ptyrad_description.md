# Task Description: PtyRAD Ptychographic Reconstruction Pipeline

## 0. Variable Taxonomy

### Naming Convention
PtyRAD uses PyTorch `nn.Parameter` for optimizable variables. Complex objects are stored as separate amplitude and phase real-valued arrays, recombined during forward pass. No systematic prefix convention detected; variables named descriptively (`object_amp`, `object_phase`, `probe`, `positions`, `tilts`, `thickness`).

### Primal Variables

| name | shape | dtype | role | requires_grad condition | onset | description |
|------|-------|-------|------|------------------------|-------|-------------|
| `object_amp` | `(N, Nx_O, Ny_O, Nz_O)` | float32 | Object amplitude per mode per slice | `lr_obj_amp > 0 and iter >= onset_obj_amp` | configurable | Initialized to 1.0 (flat amplitude) |
| `object_phase` | `(N, Nx_O, Ny_O, Nz_O)` | float32 | Object phase per mode per slice | `lr_obj_phase > 0 and iter >= onset_obj_phase` | configurable | Initialized to uniform random in [0, 1e-8] |
| `probe` | `(M, Nx_P, Ny_P, 2)` | float32 | Complex probe modes (real+imag trailing dim) | `lr_probe > 0 and iter >= onset_probe` | configurable | Initialized from aberration function or imported |
| `positions` | `(N_tot, 2)` | float32 | Probe scan positions (x,y) in pixels | `lr_pos > 0 and iter >= onset_pos` | configurable | From scan metadata |
| `tilts` | `(N_tot, 2)` | float32 | Local specimen tilt (θ_x, θ_y) per position | `lr_tilt > 0 and iter >= onset_tilt` | configurable | Initialized to 0 or global estimate |
| `thickness` | `(1,)` | float32 | Slice thickness Δz | `lr_thickness > 0 and iter >= onset_thickness` | configurable | From estimated sample thickness / Nz_O |

### Model Buffers

| name | shape | dtype | role | description |
|------|-------|-------|------|-------------|
| `kx_grid` | `(Nx_P,)` | float32 | Reciprocal-space freq grid x | From `torch.fft.fftfreq(Nx_P, d=pixel_size)` |
| `ky_grid` | `(Ny_P,)` | float32 | Reciprocal-space freq grid y | From `torch.fft.fftfreq(Ny_P, d=pixel_size)` |
| `k_sq` | `(Nx_P, Ny_P)` | float32 | |k|² grid for propagator | `kx_grid[:,None]² + ky_grid[None,:]²` |
| `wavelength` | `()` | float32 | Electron wavelength λ | From accelerating voltage |
| `probe_mask` | `(Nx_P, Ny_P)` | float32 | Fourier-space cutoff mask for probe constraint | Binary mask from semi-convergence angle |

### Observations

| name | shape | dtype | description |
|------|-------|-------|-------------|
| `I_meas` | `(N_tot, Nx_P, Ny_P)` | float32 | Measured diffraction patterns |
| `I_pacbed` | `(Nx_P, Ny_P)` | float32 | Position-averaged CBED (optional) |

### Derived Variables

| name | shape | dtype | lifetime | description |
|------|-------|-------|----------|-------------|
| `object_complex` | `(N, Nx_O, Ny_O, Nz_O)` complex64 via `amp * exp(i*phase)` | per-iteration | Complex transmission function |
| `crop_indices` | `(batch, 2)` int | per-batch | Integer pixel crop positions = `round(positions)` |
| `sub_px_shift` | `(batch, 2)` float32 | per-batch | `positions - round(positions)` |
| `O_j` | `(batch, N, Nx_P, Ny_P, Nz_O)` complex64 | per-batch | Cropped object patches |
| `P_j` | `(batch, M, Nx_P, Ny_P)` complex64 | per-batch | Sub-pixel shifted probes |
| `psi_j` | `(batch, M, N, Nx_P, Ny_P)` complex64 | per-batch | Exit waves |
| `I_model` | `(batch, Nx_P, Ny_P)` float32 | per-batch | Modeled diffraction intensities |
| `propagator` | `(batch, Nx_P, Ny_P)` complex64 | per-batch or cached | Multislice propagator M(θ_j, Δz) |

### Dual Variables
Not applicable — solver is first-order gradient descent (Adam).

### Config Parameters

| name | type | default | description |
|------|------|---------|-------------|
| `batch_size` | int | dataset-dependent | Mini-batch size per iteration |
| `n_iterations` | int | ~100–300 | Total iterations |
| `lr_obj_amp` | float | ~1e-3 | Learning rate for object amplitude |
| `lr_obj_phase` | float | ~1e-2 | Learning rate for object phase |
| `lr_probe` | float | ~1e-1 | Learning rate for probe |
| `lr_pos` | float | ~1e-3 | Learning rate for positions |
| `lr_tilt` | float | ~0 | Learning rate for tilts |
| `lr_thickness` | float | ~0 | Learning rate for thickness |
| `loss_type` | str | "gaussian" | "gaussian" or "poisson" |
| `p_power` | float | 0.5 (Gaussian), 1.0 (Poisson) | Power applied to intensities in loss |
| `w1,w2,w3,w4` | float | 1,0,0,0 | Loss term weights |
| `M` | int | 1 | Number of probe modes |
| `N` | int | 1 | Number of object modes |
| `Nz_O` | int | 1 | Number of object slices |
| `p_sparse` | float | 1.0 | Lp exponent for sparsity regularization |
| `epsilon` | float | 1e-6 | Numerical stability constant |

## 1. Data Preprocessing

### Problem Formulation
Electron ptychography inverse problem: recover complex object O (potentially multislice), probe P (potentially mixed-state), scan positions ρ, specimen tilts θ, and slice thickness Δz from a set of measured diffraction intensity patterns. The forward model is a mixed-state multislice ptychography model optimized via AD-based gradient descent.

### Preprocessing Steps
1. Load 4D-STEM data from supported formats (raw, hdf5, mat, tif, npy)
2. Apply geometric transforms: permutation, reshaping, flipping, transposing, cropping as specified in config
3. Optionally pad/resample diffraction patterns (can be done on-the-fly to save GPU memory)
4. Load or compute initial probe from aberration function parameters (defocus, semi-convergence angle, voltage)
5. Load scan positions from metadata; convert to pixel units using real-space pixel size
6. Initialize object: amplitude = 1.0 everywhere; phase = uniform random in [0, 1e-8]
7. Initialize tilts to zeros (or global estimate), thickness from estimated specimen thickness / Nz_O
8. Transfer all tensors to GPU as PyTorch tensors

## 2. Forward Model

### Subroutine: `build_object_complex`
- **Input:** `object_amp` (N, Nx_O, Ny_O, Nz_O), `object_phase` (N, Nx_O, Ny_O, Nz_O)
- **Output:** `object_complex` (N, Nx_O, Ny_O, Nz_O) complex
- **Pseudocode:** `object_complex = object_amp * exp(i * object_phase)`

### Subroutine: `crop_object_patch`
- **Input:** `object_complex`, batch positions
- **Output:** `O_j` (batch, N, Nx_P, Ny_P, Nz_O) complex
- **Pseudocode:** Compute `crop_indices = round(positions[batch])`. For each position j in batch, crop Nx_P × Ny_P patch centered at crop_indices[j] from each object mode and slice.

### Subroutine: `shift_probe`
- **Input:** `probe` (M, Nx_P, Ny_P) complex, `sub_px_shift` (batch, 2)
- **Output:** `P_j` (batch, M, Nx_P, Ny_P) complex
- **Pseudocode:**
```
sub_px_shift = positions[batch] - round(positions[batch])
For each j in batch:
    phase_ramp = exp(-2πi * (kx * Δrx_j + ky * Δry_j))
    P_j[j] = IFFT2(FFT2(probe) * phase_ramp)
```

### Subroutine: `compute_propagator`
- **Input:** `tilts` (batch, 2), `thickness` scalar, `k_sq`, `kx_grid`, `ky_grid`, `wavelength`
- **Output:** `propagator` (batch, Nx_P, Ny_P) complex
- **Pseudocode (Eq. 4):**
```
M(k) = exp(-iπλ|k|²Δz + 2πiΔz(kx*tan(θx) + ky*tan(θy)))
```

### Subroutine: `multislice_forward`
- **Input:** `P_j` (batch, M, Nx_P, Ny_P), `O_j` (batch, N, Nx_P, Ny_P, Nz_O), `propagator`
- **Output:** `psi_j` (batch, M, N, Nx_P, Ny_P) complex — exit waves
- **Pseudocode (Eq. 3):**
```
For each (m, n) mode pair:
    wave = P_j[:, m] * O_j[:, n, :, :, 0]   # first slice transmission
    For s in range(1, Nz_O):
        wave = IFFT2(FFT2(wave) * propagator)  # propagate
        wave = wave * O_j[:, n, :, :, s]       # next slice transmission
    psi_j[:, m, n] = wave
```

### Subroutine: `compute_intensity`
- **Input:** `psi_j` (batch, M, N, Nx_P, Ny_P) complex
- **Output:** `I_model` (batch, Nx_P, Ny_P) float
- **Pseudocode (Eq. 2):**
```
I_model = sum over m,n of |FFT2(psi_j[:, m, n])|²
```

### Conditional Operator Paths

| Condition | Path |
|-----------|------|
| `Nz_O == 1` | Single-slice: skip propagation loop, exit wave = P_j * O_j |
| `Nz_O > 1` | Multislice: full propagation loop |
| `M == 1, N == 1` | No mixed-state summation needed |
| `global_tilt` | Single tilt value for all positions |
| `local_tilt` | Per-position tilt values |

## 3. Inversion

### Objective Function

**Full expression (Eq. 8):**
$$\mathcal{L}_{\text{total}} = w_1 \mathcal{L}_{\text{Gaussian}} + w_2 \mathcal{L}_{\text{Poisson}} + w_3 \mathcal{L}_{\text{PACBED}} + w_4 \mathcal{L}_{\text{sparse}}$$

**Data fidelity term (Gaussian, Eq. 5):**
$$\mathcal{L}_{\text{Gaussian}} = \frac{\sqrt{\langle (I_{\text{model}}^p - I_{\text{meas}}^p)^2 \rangle_{\mathcal{D},\mathcal{B}}}}{\langle I_{\text{meas}}^p \rangle_{\mathcal{D},\mathcal{B}}}$$
Default p=0.5. Averaging is over detector pixels (D) and batch (B).

**Regularization Terms:**

| name | formula | target | weight | default_weight | Lp |
|------|---------|--------|--------|---------------|-----|
| Poisson NLL | Eq. 6 | I_model vs I_meas | w2 | 0 | p=1 |
| PACBED | Eq. 7 | batch-averaged I_model vs I_meas | w3 | 0 | p=0.5 |
| Sparsity | `⟨|O_phase|^p⟩^(1/p)` over R,B | object_phase | w4 | 0 | p=1 (L1) |

**Constraints Table:**

| name | target | operation | interval | relaxation | description |
|------|--------|-----------|----------|------------|-------------|
| Probe orthogonalization | probe modes | SVD-based orthogonalization (Thibault & Menzel 2013) | every K iters | N/A | Decorrelate mixed probe modes |
| Fourier cutoff mask | probe | Multiply FFT(probe) by binary aperture mask | configurable | N/A | Enforce aperture limit |
| Object blur (real-space) | object | Gaussian convolution in real space | configurable | configurable | Suppress high-freq noise in multislice |
| Object blur (reciprocal) | object | Low-pass filter in Fourier domain | configurable | configurable | Stability for multislice |
| Amplitude thresholding | object_amp | Clip to [min_val, max_val] | configurable | N/A | Physical amplitude bounds |
| Phase positivity | object_phase | max(phase, 0) or related | configurable | N/A | Enforce non-negative phase for thin specimens |
| Complex relation | object | Enforce amp-phase relationship | configurable | relaxation factor | Physical consistency |

### Solver/Optimization

**Optimizer:** Adam (default). All 14 PyTorch optimizers supported. Separate learning rate per parameter group.

**Per-iteration step sequence:**
1. **Parameter activation check:** For each parameter, set `requires_grad = (lr > 0) and (iter >= onset)`
2. **Batch sampling:** Sample mini-batch of indices from N_tot positions (shuffled each epoch)
3. **Forward pass:** `build_object_complex` → `crop_object_patch` → `shift_probe` → `compute_propagator` → `multislice_forward` → `compute_intensity`
4. **Retrieve measurements:** Index `I_meas[batch]`
5. **Loss computation:** Compute L_total per Eq. 8 with active weights
6. **`loss.backward()`:** AD computes gradients for all active parameters
7. **`optimizer.step()`:** Adam update for all parameter groups
8. **`optimizer.zero_grad()`**
9. **Constraint application (post-step):** Apply all active physical constraints to `.data` of relevant parameters
10. **Logging:** Record loss, optionally compute diagnostics

**One iteration = one full pass through all N_tot patterns (multiple mini-batches).**

### Depth Regularization
The paper proposes a real-space depth regularization that avoids wrap-around artifacts, useful for twisted 2D materials and vertical heterostructures. This applies blurring/smoothing in the z-direction of the object to regularize depth reconstruction. [RECONSTRUCTED_FROM_PAPER_TEXT: exact formula not provided in available text; implementation follows the constraint application framework]

### Numerical Stability
- ε = 1e-6 added inside log in Poisson loss
- Intensities raised to power p (default 0.5 for Gaussian) to compress dynamic range

### JSON Output Schema

```json
{
  "data_preprocessing": {
    "diffraction_patterns_shape": {"type": "array", "shape": "[N_tot, Nx_P, Ny_P]"},
    "positions_shape": {"type": "array", "shape": "[N_tot, 2]"},
    "pixel_size_angstrom": {"type": "float"}
  },
  "optimization": {
    "processing_time_total": {"type": "float", "unit": "seconds"},
    "iterations_completed": {"type": "int"},
    "final_loss": {"type": "float"},
    "loss_curve": {"type": "array", "shape": "[n_iterations]"},
    "per_iteration_time_avg": {"type": "float", "unit": "seconds"}
  },
  "reconstructed_variables": {
    "object_amplitude": {"type": "array", "shape": "[N, Nx_O, Ny_O, Nz_O]", "dtype": "float32"},
    "object_phase": {"type": "array", "shape": "[N, Nx_O, Ny_O, Nz_O]", "dtype": "float32"},
    "probe": {"type": "array", "shape": "[M, Nx_P, Ny_P, 2]", "dtype": "float32", "note": "real+imag"},
    "positions_refined": {"type": "array", "shape": "[N_tot, 2]", "dtype": "float32"},
    "tilts_refined": {"type": "array", "shape": "[N_tot, 2]", "dtype": "float32"},
    "thickness_refined": {"type": "float", "dtype": "float32"}
  },
  "evaluation": {
    "depth_summed_phase": {"type": "array", "shape": "[Nx_O, Ny_O]", "dtype": "float32", "note": "sum over slices of object_phase[0]"},
    "fft_power_spectrum": {"type": "array", "shape": "[Nx_O, Ny_O]", "dtype": "float32"}
  },
  "optional_analysis": {
    "ssim_vs_ground_truth": {"type": "float", "note": "requires ground truth"},
    "information_limit_pm": {"type": "float", "note": "from FFT analysis"}
  }
}
```

## 4. Evaluation

### Quality Metrics

| Metric | Formula/Reference | Notes |
|--------|------------------|-------|
| SSIM | `skimage.metrics.structural_similarity` or Wang et al. (2004) | Computed on depth-summed 2D phase images; subtract minimum phase before comparison |
| Loss (L_total) | Eq. 8 | Track per iteration |
| FFT power spectrum | `|FFT2(depth_summed_phase)|²` | Log-scale visualization; assess information limit |
| Information limit | Radial frequency at which FFT signal drops to noise floor | Measured in pm from FFT ring analysis |

### Solver-Level Diagnostics

| Diagnostic | Description |
|------------|-------------|
| Total reconstruction time | Wall-clock excluding init and save |
| Per-iteration time | Average over all iterations |
| Final loss value | L_total at last iteration |
| Convergence curve | Loss vs. iteration (and optionally vs. wall-clock time) |
| Per-variable stats | Mean, std, min, max, shape for object_amp, object_phase, probe |
| SSIM curve | SSIM vs. iteration (when ground truth available) |

### Diagnostic Outputs
- **Depth-summed phase image:** Sum `object_phase` over slice dimension for each object mode
- **FFT power spectrum:** 2D FFT of depth-summed phase, displayed in log scale
- **Per-slice phase images:** Individual slice reconstructions for multislice
- **Probe modes:** Amplitude and phase of each probe mode
- **Position scatter plot:** Refined vs. initial positions
- **Convergence curve:** Loss and/or SSIM vs. iteration

### Implementation Cautions

1. **Complex storage:** Object is stored as separate real-valued amplitude and phase arrays, NOT as complex tensors. Complex combination `amp * exp(i*phase)` happens only during forward pass. This enables separate learning rates.

2. **Sub-pixel shifting:** Probe shift for sub-pixel positioning is performed via Fourier-domain phase ramp multiplication, NOT spatial interpolation. This requires consistent FFT grid conventions.

3. **Batch processing:** One "iteration" = full pass over all N_tot patterns in mini-batches. The batch dimension is the first dimension during forward computation. Shuffling occurs per epoch.

4. **Propagator recomputation:** When tilts are optimizable and local (per-position), the propagator must be recomputed per batch element. When global or fixed, it can be cached.

5. **Constraint application timing:** All physical constraints are applied AFTER `optimizer.step()`, directly modifying `.data` attributes. They do NOT participate in the computational graph.

6. **Onset scheduling:** Parameters should be activated gradually (pyramidal approach). Activating all parameters simultaneously can destabilize optimization. Typical order: object_phase first → probe → positions → tilts/thickness later.

7. **Power transform:** The intensity power p=0.5 for Gaussian loss effectively computes loss on amplitude rather than intensity, which compresses dynamic range and improves convergence for high-dynamic-range data.

8. **Normalization of loss:** Both Gaussian and Poisson losses are normalized by `⟨I_meas^p⟩` to make the loss scale-invariant across datasets with different total counts.

9. **Multi-GPU:** PtyRAD supports multi-GPU via PyTorch distributed (Accelerate library by Hugging Face). Batch is split across GPUs.

10. **Hyperparameter tuning:** Optuna-based Bayesian optimization wraps the entire reconstruction loop, optimizing non-AD parameters (batch size, learning rates, defocus, semi-convergence angle, etc.) by evaluating reconstruction quality metrics.