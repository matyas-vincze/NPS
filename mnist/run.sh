#!/bin/bash
#python3 ./NPS-code-to-run/temp/main.py
python3 ./NPS-code-to-run/main.py --cuda=True --n_epochs=150 --dropout_prob=0.1 --batch_size=128 --load_model=False --wandb=True --seed=1
# python3 ./testing.py --load_model=True --cuda=False --batch_size=64