# Delete only the bad backbone/head files (trained with random sampling)
!rm -f /content/drive/MyDrive/VELM_checkpoints/backbone_grad.eqx
!rm -f /content/drive/MyDrive/VELM_checkpoints/energy_head_grad.eqx
!rm -f /content/drive/MyDrive/VELM_checkpoints/backbone_eggroll*.eqx
!rm -f /content/drive/MyDrive/VELM_checkpoints/energy_head_eggroll*.eqx

# Keep these — they're good:
# calm_ae_best.eqx      ← 99.9% AE (independent chunks, no sequential bug)
# calm_ae_best.json
# calm_ae_final.eqx
# teacher_vectors.npy   ← teacher hidden states per chunk (also independent)