# exp01_fmri_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** subj01
- **Modelo:** fMRIEncoder(hidden=4096, out=2048) + CLIPHead
- **Config:** `outputs\exp01_fmri_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.283 / 0.609 / 0.722
- mean cosine: 0.521

## test
- retrieval Top-1/5/10: 0.491 / 0.874 / 0.950
- mean cosine: 0.517

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.