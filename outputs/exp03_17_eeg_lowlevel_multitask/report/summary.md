# exp03_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.005 / 0.034 / 0.047
- mean cosine: 0.505
- low-level mean Pearson r: 0.012

## test
- retrieval Top-1/5/10: 0.005 / 0.050 / 0.105
- mean cosine: 0.469
- low-level mean Pearson r: 0.011

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.