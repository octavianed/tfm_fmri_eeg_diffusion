# exp03_raw_baseline_minus100_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_baseline_minus100_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.004 / 0.019 / 0.036
- mean cosine: 0.365
- low-level mean Pearson r: 0.010

## test
- retrieval Top-1/5/10: 0.010 / 0.045 / 0.100
- mean cosine: 0.376
- low-level mean Pearson r: 0.013

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.