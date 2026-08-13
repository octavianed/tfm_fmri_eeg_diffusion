# exp03_raw_frequency_0_5_40_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_frequency_0_5_40_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.002 / 0.014 / 0.022
- mean cosine: 0.284
- low-level mean Pearson r: 0.006

## test
- retrieval Top-1/5/10: 0.010 / 0.040 / 0.065
- mean cosine: 0.267
- low-level mean Pearson r: 0.009

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.