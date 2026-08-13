# exp01_raw_sampling_100hz_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_sampling_100hz_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.010 / 0.039 / 0.059
- mean cosine: 0.340

## test
- retrieval Top-1/5/10: 0.020 / 0.070 / 0.105
- mean cosine: 0.327

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.