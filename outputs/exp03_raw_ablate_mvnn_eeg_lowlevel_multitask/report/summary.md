# exp03_raw_ablate_mvnn_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_ablate_mvnn_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.002 / 0.011 / 0.018
- mean cosine: 0.616
- low-level mean Pearson r: 0.004

## test
- retrieval Top-1/5/10: 0.015 / 0.080 / 0.170
- mean cosine: 0.642
- low-level mean Pearson r: 0.005

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.