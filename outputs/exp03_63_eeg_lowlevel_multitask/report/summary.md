# exp03_63_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_63_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.005 / 0.031 / 0.051
- mean cosine: 0.489
- low-level mean Pearson r: 0.011

## test
- retrieval Top-1/5/10: 0.005 / 0.085 / 0.160
- mean cosine: 0.450
- low-level mean Pearson r: 0.017

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.