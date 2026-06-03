# Post-training Ctrl-World evaluation

Matched-seed comparison using the same snapshots, prompts, guidance, and rollout length within each task. Only the policy checkpoint changes.

## Cube

Totals: base strict 4/10; base lax 7/10; base g3 baseline 4/10; base g5 baseline 8/10; Ctrl-World LoRA 10/10; teleop LoRA 2/10.

| val_id | base (chat: strict/lax) | base g3 baseline | base g5 baseline | Ctrl-World LoRA | teleop LoRA |
| --- | --- | --- | --- | --- | --- |
| 0002 | yes strict / yes lax | yes | yes | yes | no |
| 0007 | no strict / no lax | no | no | yes | no |
| 0008 | no strict / yes lax | no | yes | yes | no |
| 0009 | yes strict / yes lax | no | yes | yes | no |
| 0015 | no strict / yes lax | no | yes | yes | no |
| 0016 | no strict / no lax | yes | yes | yes | yes |
| 0018 | no strict / yes lax | yes | yes | yes | no |
| 0041 | yes strict / yes lax | no | no | yes | no |
| 0048 | yes strict / yes lax | no | yes | yes | yes |
| 0057 | no strict / no lax | yes | yes | yes | no |

## Tomato

Totals: base 3/10; Ctrl-World LoRA 10/10; teleop LoRA 7/10.

| val_id | base | Ctrl-World LoRA | teleop LoRA |
| --- | --- | --- | --- |
| 0003 | yes | yes | yes |
| 0005 | no | yes | yes |
| 0035 | no | yes | yes |
| 0042 | no | yes | yes |
| 0044 | no | yes | yes |
| 0063 | yes | yes | no |
| 0065 | no | yes | yes |
| 0067 | no | yes | yes |
| 0085 | no | yes | no |
| 0090 | yes | yes | no |
