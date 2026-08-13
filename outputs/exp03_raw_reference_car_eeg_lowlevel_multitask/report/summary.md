# exp03_raw_reference_car_eeg_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_raw_reference_car_eeg_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.004 / 0.025 / 0.038
- mean cosine: 0.335
- low-level mean Pearson r: 0.008

## test
- retrieval Top-1/5/10: 0.010 / 0.065 / 0.120
- mean cosine: 0.320
- low-level mean Pearson r: 0.016

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.