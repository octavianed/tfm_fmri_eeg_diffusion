# exp03_raw_sampling_100hz_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_sampling_100hz_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.010 / 0.037 / 0.057
- mean cosine: 0.396
- low-level mean Pearson r: 0.012

## test
- retrieval Top-1/5/10: 0.020 / 0.070 / 0.140
- mean cosine: 0.362
- low-level mean Pearson r: 0.010

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.