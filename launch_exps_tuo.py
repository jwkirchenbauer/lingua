# fmt: off
import os
from itertools import product, chain

# LIST_CFGS = True
LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

LAUNCHER_FILEPATH = "/p/vast1/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = (
    "/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib"
)

ROCM_VERSION = "6.3.0"
RCCL_CFG = "rdzv-lbann"

# EXTRA_COMPILE_FLAGS = False
EXTRA_COMPILE_FLAGS = True

# LOG_RECOMPILES=False
LOG_RECOMPILES = True

# QOS = "pdebug"
QOS = "pbatch"
BANK = "effml"
TIME_LIMIT = 29
REPETITIONS = 1
DEPENDENCY = None

# QOS = "pbatch"
# BANK = "effml"
# # BANK = "guard"
# TIME_LIMIT = 1440
# REPETITIONS = 5
# DEPENDENCY = "afterany"

BASE_OUT_DIR = f"/p/vast1/kirchenb/singleshot-root/lingua/outputs"

BASE_RUN_NAME = f"test"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

# INDUCTOR_CACHE=None
INDUCTOR_CACHE="/l/ssd/$USER"

GPN = 4

# Cfgs
exp_list = [
    # ["apps.main.train", "apps/main/configs/llama_7B_fw_10bt_tuo.yaml", 1, 1, 2, 4096],
    # ["apps.main.train", "apps/main/configs/llama_7B_fw_10bt_tuo.yaml", 1, GPN, 2, 4096],
    ["apps.main.train", "apps/main/configs/llama_7B_fw_10bt_tuo.yaml", 1, GPN, 2, 4096, "sdpa"],
    ["apps.main.train", "apps/main/configs/llama_7B_fw_10bt_tuo.yaml", 1, GPN, 2, 4096, "flex_attention"],
    ["apps.main.train", "apps/main/configs/llama_7B_fw_10bt_tuo.yaml", 1, GPN, 2, 2048, "flex_attention"],
]


# sweep_hparam = [
#     [
#     "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/tinystories_gpt2/tiny_train_*.bin",
#     "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/tinystories_gpt2/tiny_validation_*.bin",
#     473992006, # toks in train
#     ],
# ]
# exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))


final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        cfg,
        nodes,
        gpn,
        mbsz,
        slen,
        attn,
        
    ) = exp

    gpus = nodes * gpn

    cli_args = ""

    cfg_str = cfg.split("/")[-1].replace(".yaml","")
    cli_args += f" config={cfg}"

    bsz_str = f"mb{mbsz}-sl{slen}"
    cli_args += f" data.batch_size={mbsz} data.seq_len={slen}"

    # attn control
    attn_str = f"attn{attn}"
    cli_args += f" attn_impl={attn}"

    # mod more things
    # ...

    # join to a unique run name for the experiment
    run_name = (
        f"{BASE_RUN_NAME}_{cfg_str}_{attn_str}_{bsz_str}_{nodes}N{gpus}n"
    )

    # put together the actual "train.py" command
    custom_invocation = f"python -u -m {script} {cli_args}"

    # make the complete launcher command
    command = f"""\
    python {LAUNCHER_FILEPATH} \
        --output_dir={BASE_OUT_DIR}/{BASE_RUN_NAME} \
        --wandb_offline={WANDB_OFFLINE} \
        --rocm_version={ROCM_VERSION} \
        --rccl_installdir={RCCL_INSTALL_DIR} \
        --rccl_cfg={RCCL_CFG} \
        --cache_dir={INDUCTOR_CACHE} \
        --qos={QOS} \
        --bank={BANK} \
        --repetitions={REPETITIONS}{f' --dependency={DEPENDENCY}' if DEPENDENCY is not None else ''} \
        --minutes={TIME_LIMIT} \
        --nodes={nodes} \
        --gpus_per_node={gpn} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation}' \
        --pass_run_name=False \
        --add_compile_flags={EXTRA_COMPILE_FLAGS} \
        --log_recompiles={LOG_RECOMPILES} \
        {'--dryrun' if WRITE_ONLY else ''}
    """
    total_launches += 1
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        # print(command)

print(f"Total launches: {total_launches}")
