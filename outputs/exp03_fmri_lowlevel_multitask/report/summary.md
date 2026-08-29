# exp03_fmri_lowlevel_multitask — summary

- **Objetivo:** CLIP + low-level multitask
- **Sujetos:** subj01
- **Modelo:** fMRIEncoder(hidden=4096, out=2048) + CLIPHead + LowLevelHead
- **Config:** `outputs\exp03_fmri_lowlevel_multitask\config.yaml`

## val
- retrieval Top-1/5/10: 0.290 / 0.613 / 0.733
- mean cosine: 0.511
- low-level mean Pearson r: 0.047

## test
- retrieval Top-1/5/10: 0.478 / 0.849 / 0.956
- mean cosine: 0.508
- low-level mean Pearson r: 0.050

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.