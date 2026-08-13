# exp03_raw_baseline_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_baseline_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.004 / 0.023 / 0.039
- mean cosine: 0.381
- low-level mean Pearson r: 0.009

## test
- retrieval Top-1/5/10: 0.005 / 0.045 / 0.115
- mean cosine: 0.366
- low-level mean Pearson r: 0.014

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.