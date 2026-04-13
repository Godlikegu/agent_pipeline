# Final Task Execution Report (No Skills Injection)

**Model:** Vendor2/Claude-4.6-opus  
**Skills:** Disabled (retrieval_enabled=false, learning_enabled=false)  
**Date:** 2026-04-07 ~ 2026-04-08  
**Total Tasks:** 51  

---

## Overall Results

| Metric | Value |
|--------|-------|
| **Total Tasks** | 51 |
| **Successful** | 18 |
| **Failed** | 33 |
| **Success Rate** | **35.3%** (18/51) |

---

## Successful Tasks (18/51)

| # | Task Name | Loops Used | Time (s) |
|---|-----------|-----------|----------|
| 1 | ct_dual_energy | 3 | 1246.4 |
| 2 | ct_sparse_view | 0 | 301.4 |
| 3 | eht_black_hole_dynamic | 4 | 3890.4 |
| 4 | eht_black_hole_feature_extraction_dynamic | 0 | 456.1 |
| 5 | eit_conductivity_reconstruction | 2 | 1021.9 |
| 6 | hessian_sim | 1 | 2192.7 |
| 7 | lucky_imaging | 2 | 1097.5 |
| 8 | mcr_hyperspectral | 0 | 424.5 |
| 9 | mri_dynamic_dce | 2 | 2046.8 |
| 10 | mri_grappa | 3 | 1975.1 |
| 11 | mri_sense | 0 | 189.8 |
| 12 | mri_t2_mapping | 0 | 423.9 |
| 13 | pet_mlem | 0 | 930.4 |
| 14 | photoacoustic_tomography | 2 | 1038.9 |
| 15 | pnp_mri_reconstruction | 1 | 736.8 |
| 16 | raman_cell_phenotyping | 3 | 488.2 |
| 17 | ultrasound_sos_tomography | 4 | 4177.9 |
| 18 | weather_radar_data_assimilation | 0 | 268.9 |

---

## Failed Tasks (33/51)

| # | Task Name | Loops Used | Time (s) | Error |
|---|-----------|-----------|----------|-------|
| 1 | SSNP_ODT | 5 | 7450.3 | Workflow returned False |
| 2 | confocal-nlos-fk | 5 | 2612.3 | Workflow returned False |
| 3 | conventional_ptychography | 5 | 2174.8 | Workflow returned False |
| 4 | ct_fan_beam | 5 | 2407.1 | Workflow returned False |
| 5 | ct_poisson_lowdose | 5 | 2673.5 | Workflow returned False |
| 6 | differentiable_deflectometry | 5 | 5841.0 | Workflow returned False |
| 7 | diffusion_mri_dti | 5 | 723.5 | Workflow returned False |
| 8 | eht_black_hole_UQ | 5 | 3136.2 | Workflow returned False |
| 9 | eht_black_hole_original | 5 | 3256.4 | Workflow returned False |
| 10 | eht_black_hole_tomography | 5 | 8076.8 | Workflow returned False |
| 11 | electron_ptychography | 5 | 1847.6 | Workflow returned False |
| 12 | exoplanet_imaging | 5 | 807.6 | Workflow returned False |
| 13 | fourier_ptychography | 0 | 13879.0 | Request timed out |
| 14 | fpm_inr_reconstruction | 5 | 2969.7 | Workflow returned False |
| 15 | insar_phase_unwrapping | 5 | 5441.8 | Workflow returned False |
| 16 | lensless_imaging | 5 | 9335.0 | Workflow returned False |
| 17 | light_field_microscope | 5 | 8490.8 | Workflow returned False |
| 18 | microscope_denoising | 5 | 719.6 | Workflow returned False |
| 19 | mri_l1_wavelet | 5 | 2491.1 | Workflow returned False |
| 20 | mri_noncartesian_cs | 5 | 2912.5 | Workflow returned False |
| 21 | mri_pnp_admm | 5 | 2241.1 | Workflow returned False |
| 22 | mri_tv | 5 | 2363.6 | Workflow returned False |
| 23 | mri_varnet | 5 | 1268.0 | Workflow returned False |
| 24 | plane_wave_ultrasound | 5 | 2338.3 | Workflow returned False |
| 25 | reflection_ODT | 5 | 6651.5 | Workflow returned False |
| 26 | seismic_FWI_original | 5 | 6562.2 | Workflow returned False |
| 27 | shapelet_source_reconstruction | 5 | 2899.9 | Workflow returned False |
| 28 | single_molecule_light_field | 5 | 3744.1 | Workflow returned False |
| 29 | spectral_snapshot_compressive_imaging | 5 | 10138.3 | Workflow returned False |
| 30 | usct_FWI | 5 | 14174.5 | Workflow returned False |
| 31 | xray_laminography_tike | 5 | 5644.2 | Workflow returned False |
| 32 | xray_ptychography_tike | 5 | 1074.6 | Workflow returned False |
| 33 | xray_tooth_gridrec | 5 | 1297.0 | Workflow returned False |

---

## Summary by Category

| Domain | Total | Success | Failed | Rate |
|--------|-------|---------|--------|------|
| CT (computed tomography) | 4 | 2 | 2 | 50.0% |
| MRI | 9 | 4 | 5 | 44.4% |
| EHT (black hole) | 5 | 2 | 3 | 40.0% |
| Ptychography | 4 | 0 | 4 | 0.0% |
| Ultrasound | 3 | 2 | 1 | 66.7% |
| Optical/Microscopy | 6 | 2 | 4 | 33.3% |
| Other imaging | 20 | 6 | 14 | 30.0% |

---

## Conclusion

Without skills injection, the pipeline achieves a **35.3% success rate (18/51 tasks)** using the Vendor2/Claude-4.6-opus model. Most failures exhaust all 5 retry iterations, indicating the model struggles with complex scientific reconstruction tasks without domain-specific guidance. Tasks that succeed tend to be those with more straightforward algorithms (e.g., FBP-based CT, basic MRI, simple denoising/mapping).
