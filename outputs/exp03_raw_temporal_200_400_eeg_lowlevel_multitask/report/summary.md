# exp03_raw_temporal_200_400_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_temporal_200_400_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.007 / 0.027 / 0.039
- mean cosine: 0.331
- low-level mean Pearson r: 0.007

## test
- retrieval Top-1/5/10: 0.050 / 0.120 / 0.175
- mean cosine: 0.307
- low-level mean Pearson r: 0.014

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.