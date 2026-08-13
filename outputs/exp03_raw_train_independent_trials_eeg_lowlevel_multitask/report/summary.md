# exp03_raw_train_independent_trials_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_train_independent_trials_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.005 / 0.018 / 0.030
- mean cosine: 0.480
- low-level mean Pearson r: 0.010

## test
- retrieval Top-1/5/10: 0.010 / 0.030 / 0.065
- mean cosine: 0.464
- low-level mean Pearson r: 0.008

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.