# exp03_raw_temporal_100_600_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_temporal_100_600_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.003 / 0.024 / 0.041
- mean cosine: 0.550
- low-level mean Pearson r: 0.011

## test
- retrieval Top-1/5/10: 0.030 / 0.105 / 0.195
- mean cosine: 0.553
- low-level mean Pearson r: 0.013

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.