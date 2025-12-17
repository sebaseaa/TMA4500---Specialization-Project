'''
Train time a stepping DeepONet for the 2D wave equation.
'''

import os
import deepxde as dde
from time_stepping_utils import *


def train_step_operator(step_idx: int, trained_tags_so_far=None) -> str:
    '''
    Train DeepONet for operator step step_idx -> step_idx+1.
    step_idx == 0: sample from GRF.
    step_idx >= 1: data generated via GRF and all previously trained operators.
    '''
    if trained_tags_so_far is None:
        trained_tags_so_far = []

    print(
        f"\n Training operator for step {step_idx} -> {step_idx+1}",
        flush=True,
    )

    if step_idx == 0:
        data = make_dataset(0)
    else:
        data = make_dataset(step_idx, trained_tags_before_this_stage=trained_tags_so_far)

    net = make_net()
    model = dde.Model(data, net)

    if step_idx == 0:
        loss_weights = loss_weights_base
    else:
        loss_weights = [ru_weight_stage_after_1 * loss_weights_base[0]] + \
                       loss_weights_base[1:]

    model.compile(
        "adam",
        lr=lr,
        loss_weights=loss_weights,
        decay=("lambda", warmup_cosine),
    )
    model.train(iterations=1000 + 9000, display_every=100)  
    dde.optimizers.LBFGS_options["iter_per_step"] = 100
    dde.optimizers.set_LBFGS_options(maxiter=l_bfgs_steps)
    model.compile("L-BFGS", loss_weights=loss_weights)
    model.train()

    tag = f"{save_prefix}_{step_idx:02d}_to_{step_idx+1:02d}"
    save_path = os.path.join(save_dir, tag)
    model.save(save_path)
    print(f"[SAVED] {save_path}", flush=True)

    return tag


tags = []  

for t in range(n_steps):
    tag = train_step_operator(
        t,
        trained_tags_so_far=tags
    )
    tags.append(tag)

print("Training complete.")
