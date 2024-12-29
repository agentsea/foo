# Training

## Molmo

```bash
torchrun --nproc-per-node $NUM_GPUS train.py
```

It seems to be working atm with at least 4 A100 80GB. 


* This is borrowed from tide-ui *