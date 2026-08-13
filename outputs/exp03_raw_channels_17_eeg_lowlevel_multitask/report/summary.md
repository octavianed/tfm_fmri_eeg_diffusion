# exp03_raw_channels_17_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_channels_17_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.005 / 0.022 / 0.033
- mean cosine: 0.463
- low-level mean Pearson r: 0.006

## test
- retrieval Top-1/5/10: 0.005 / 0.080 / 0.145
- mean cosine: 0.423
- low-level mean Pearson r: 0.014

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.