import numpy as np
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import re
import os

# import torch

from lingua.args import dataclass_from_dict
from apps.main.eval import (
    # launch_eval,
    launch_consolidate_only,
    EVAL_FOLDER_NAME,
    EvalArgs,
)
from lingua.checkpoint import RE_FOLDER, _get_key_step


args_eval = {
    "harness": {
        "tasks": [
            "hellaswag",
            {"task": "boolq", "dataset_kwargs": {"trust_remote_code": True}},
            "piqa",
            {"task": "social_iqa", "dataset_kwargs": {"trust_remote_code": True}},
            "winogrande",
            "openbookqa",
            "arc_easy",
            "arc_challenge",
            "race",
            "commonsense_qa",
            {"task": "copa", "dataset_kwargs": {"trust_remote_code": True}},
            "mmlu",
            "mmlu_pro",
        ],
    },
    "generator": {
        "max_tokens": 8192,
        "dtype": "bf16",
    },
}


def get_existing_saves(ckpt_dir):
    folders = [
        p
        for p in Path(ckpt_dir).iterdir()
        if p.is_dir() and re.match(RE_FOLDER, p.name)
    ]
    folders.sort(key=lambda p: _get_key_step(p.name))
    return folders


RUN_DUMP_DIR = f"/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_64N"
RUN_CKPTS_DIR = f"{RUN_DUMP_DIR}/checkpoints"
TRAIN_STATE_STEP = 10000


eval_args = dataclass_from_dict(EvalArgs, args_eval)

eval_args.global_step = TRAIN_STATE_STEP

# find the dir that matches the step
all_saves = get_existing_saves(RUN_CKPTS_DIR)
for save in all_saves:
    if _get_key_step(save.name) == TRAIN_STATE_STEP:
        eval_args.ckpt_dir = str(save)
        break
else:
    raise ValueError(
        f"Could not find checkpoint for step {TRAIN_STATE_STEP} in {RUN_CKPTS_DIR}"
    )


eval_args.dump_dir = str(
    os.path.join(
        RUN_DUMP_DIR,
        "evals",
        EVAL_FOLDER_NAME.format(TRAIN_STATE_STEP),
    )
)
eval_args.metric_log_dir = RUN_DUMP_DIR

launch_consolidate_only(eval_args)


# ls -la /p/vast1/pretrain/models/comma-v0.1/checkpoints/0000100000/consolidated/
# total 27354220
# drwx--S--- 2 kirchenb pretrain        4096 May  5 15:18 .
# drwx--S--- 2 kirchenb pretrain        4096 May  5 22:42 ..
# -rw------- 1 kirchenb pretrain 28010716526 May  5 15:19 model.pth
# -rw------- 1 kirchenb pretrain        3684 May  5 14:45 params.json

# python convert_lingua_weights_to_hf.py \
#     --input_dir /p/vast1/pretrain/models/comma-v0.1/checkpoints/0000100000/consolidated/ \
#     --output_dir /p/lustre5/kirchenb/common-pile-root/lingua/output/debug_conversion/hf_models/comma_v0-1_100000 \
#     --llama_version 2 \
#     --num_shards 1 \
#     --model_size 7B \
# && \
# cp /p/vast1/pretrain/models/comma-v0.1-checkpoint-hf/*tok*.json /p/lustre5/kirchenb/common-pile-root/lingua/output/debug_conversion/hf_models/comma_v0-1_100000

# the additional rename and a one liner to extract the model from the state dict allows the conversion to complete

# mv /p/lustre5/kirchenb/common-pile-root/lingua/output/debug_conversion/prod_lingua_64N_ckpts_10000/consolidated/consolidated.pth \
# /p/lustre5/kirchenb/common-pile-root/lingua/output/debug_conversion/prod_lingua_64N_ckpts_10000/consolidated/model.pth \
# && \
# python convert_lingua_weights_to_hf.py \
#     --input_dir /p/lustre5/kirchenb/common-pile-root/lingua/output/debug_conversion/prod_lingua_64N_ckpts_10000/consolidated/ \
#     --output_dir /p/lustre5/kirchenb/common-pile-root/lingua/output/debug_conversion/hf_models/prod_lingua_64N_ckpts_10000 \
#     --llama_version 2 \
#     --num_shards 1 \
#     --model_size 7B \
# && \
# cp /p/vast1/pretrain/models/comma-v0.1-checkpoint-hf/*tok*.json /p/lustre5/kirchenb/common-pile-root/lingua/output/debug_conversion/hf_models/prod_lingua_64N_ckpts_10000
