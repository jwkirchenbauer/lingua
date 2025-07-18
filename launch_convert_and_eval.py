import os
import glob
import re
import json
from itertools import product, chain
import subprocess

# DRY_RUN = True
DRY_RUN = False

WRITE_ONLY = False
# WRITE_ONLY = True

# DO_CONVERT = True
# DO_EVAL = False

DO_CONVERT = False
DO_EVAL = True

assert DO_CONVERT ^ DO_EVAL, "You must choose one of DO_CONVERT or DO_EVAL"

# EVAL_NODE_COUNT = None
EVAL_NODE_COUNT = 1

# EVAL_GPUS_PER_NODE = None
EVAL_GPUS_PER_NODE = 1

LAUNCHER_FILEPATH = "/p/lustre5/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = "/collab/usr/global/tools/rccl/$SYS_TYPE/rocm-6.3.1/install/lib"

# CONVERSION_CONDA_ENV = "/usr/workspace/wsb/kirchenb/tuolumne_conda_28_630_lingua"
# EVAL_CONDA_ENV = "/usr/workspace/wsb/kirchenb/tuolumne_conda_28_630_olmes"

WRKSPC = os.getenv("WRKSPC")

BASE_OUT_DIR = f"/p/lustre5/kirchenb/common-pile-root/lingua/output"


# QOS = "pdebug"
QOS = "pbatch"

# BANK = "guests"
BANK = "effml"

ROCM_VERSION = "6.3.0"
RCCL_MODE = "rdzv-lbann"

JOB_LIMIT = None
# JOB_LIMIT = 1

# TIME_LIMIT = 59
# TIME_LIMIT = 120
TIME_LIMIT = 960
# TIME_LIMIT = 1440

TAG_LIST = [
    "eval",
]
TAG_LIST = [str(t).replace(".", "") for t in TAG_LIST]
WANDB_TAG_STRING = f"[{','.join(TAG_LIST)}]"

# BASE_RUN_NAME = "bulk_convert_eval"
BASE_RUN_NAME = "bulk_convert_eval_corrected"

# models
exp_list = [
    # ["prod_lingua_7B_curric_64N_phase3_prelim", [120_000]],
    # ["prod_lingua_7B_curric_64N_phase3_prelim_tuo", [120_000]],
    # ["prod_lingua_64N", [80_000, 125_000]],
    # ["prod_lingua_64N", [125_000]],
    # ["prod_lingua_7B_wsd_128N_orig", [125_000]],
    # ["prod_lingua_7B_wsd_128N_1T", [80_000, 125_000]],
    # ["prod_lingua_7B_curric_64N_phase1", [42_000]],
    # ["prod_lingua_7B_curric_64N_phase2", [84_000]],
    # ["prod_lingua_7B_curric_64N_phase3", [125_000]],
    # ["prod_lingua_7B_wsd_128N", [248_000]],
    # ["prod_lingua_7B_wsd_128N", [250_000]],
    # ["prod_lingua_7B_2T_128N", [250_000]],
    # ["prod_lingua_7B_2T_long_hq_cd_128N", [246_500]],
    # ["prod_lingua_7B_2T_lin_hq_cd_128N", [230_000]],
    # ["prod_lingua_7B_2T_lin_hq_cd_128N", [239_000]],
    # ["prod_lingua_7B_2T_lin_hq_cd_128N", list(range(230_000, 240_000, 1000))],
    # this is just a hack name convention for the avg ckpt
    # ["prod_lingua_7B_2T_lin_hq_cd_128N", [239_001]],
    # the low lr run
    # ["prod_lingua_7B_2T_low_lr_lin_hq_cd_128N", [230_000]],
    ["prod_lingua_7B_2T_low_lr_lin_hq_cd_128N", list(range(231_000, 240_000, 1000))],
    ["prod_lingua_7B_2T_low_lr_lin_hq_cd_128N", [239_001]],
]

hparam = [
    # "ablation:knowledge::comma",
    # "ablation:all::comma",
    "eval:all::comma",
]
final_exp_list = list(chain(*[[exp + [hp] for hp in hparam] for exp in exp_list]))

if JOB_LIMIT is not None:
    final_exp_list = final_exp_list[:JOB_LIMIT]

for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        run_name,
        STEPS,
        eval_task_spec,
    ) = exp

    custom_invocation = ""

    parent_dir = f"{BASE_OUT_DIR}/{run_name}"
    run_ckpts_subdir = "checkpoints"
    checkpoint_dir = f"{parent_dir}/{run_ckpts_subdir}"
    hf_checkpoint_dir = f"{parent_dir}/checkpoints_hf"
    # evals_dir = f"{parent_dir}/olmes_evals"
    evals_dir = f"{parent_dir}/olmes_evals/{eval_task_spec.replace('::', '_').replace(':', '_')}"

    if not os.path.exists(parent_dir):
        continue

    # step_numbers = [step for step in STEPS]
    step_numbers = [step for step in sorted(STEPS)]
    step_numbers_10digit = [f"{int(step):010d}" for step in step_numbers]
    print(step_numbers)
    print(step_numbers_10digit)

    for step_number, step_number_10digit in zip(step_numbers, step_numbers_10digit):

        # NOTE: WARNING no gaurds here if already done

        # if custom_invocation != "":
        #     custom_invocation += "\n\n"
        if DO_CONVERT:
            # custom_invocation += f"""\
            custom_invocation = f"""\
echo "Converting checkpoint {step_number} for {run_name}"

python consolidate_ckpts.py \
--run_dump_dir {parent_dir} \
--run_ckpts_subdir checkpoints \
--train_state_step {step_number}

mv {checkpoint_dir}/{step_number_10digit}/consolidated/consolidated.pth \
{checkpoint_dir}/{step_number_10digit}/consolidated/model.pth

mkdir -p {hf_checkpoint_dir}/{step_number_10digit}

python convert_lingua_weights_to_hf.py \
--input_dir {checkpoint_dir}/{step_number_10digit}/consolidated \
--output_dir {hf_checkpoint_dir}/{step_number_10digit} \
--llama_version 2 \
--num_shards 1 \
--model_size 7B

cp /p/vast1/pretrain/models/comma-v0.1-checkpoint-hf/*tok*.json {hf_checkpoint_dir}/{step_number_10digit}

echo "Done converting {step_number} for {run_name}!"

"""
        if DO_EVAL:
            # model_args_str = r"'{" + r'"add_bos_token": true' + r"}'"
            # had to hardcode it into the library's cli defaults :/
            # custom_invocation += f"""\
            custom_invocation = f"""\
echo "Evaluating checkpoint {step_number} for {run_name}"

mkdir -p {evals_dir}/{step_number_10digit}

olmes --model {hf_checkpoint_dir}/{step_number_10digit} --task {eval_task_spec} --output-dir {evals_dir}/{step_number_10digit}

echo "Done evaluating {step_number} for {run_name}!"

"""

        if DO_CONVERT or DO_EVAL:

            command = f"""\
    python {LAUNCHER_FILEPATH} \
    --output_dir={BASE_OUT_DIR}/{BASE_RUN_NAME} \
    --rccl_installdir={RCCL_INSTALL_DIR} \
    --rccl_mode={RCCL_MODE} \
    --qos={QOS} \
    --bank={BANK} \
    --rocm_version={ROCM_VERSION} \
    --run_name={run_name}{'_convert' if DO_CONVERT else ''}{'_eval' if DO_EVAL else ''}_{step_number_10digit} \
    --nodes={EVAL_NODE_COUNT} \
    --gpus_per_node={EVAL_GPUS_PER_NODE} \
    --minutes={TIME_LIMIT} \
    --custom_invocation='{custom_invocation}' \
    --pass_run_name=False \
    {f'--dryrun' if WRITE_ONLY else ''}\
    """

            if custom_invocation == "":
                command = f'echo "Nothing to run for {run_name}"'

        total_launches += 1
        if not DRY_RUN:
            os.system(command)
        else:
            print(run_name)

print(f"Total launches: {total_launches}")
