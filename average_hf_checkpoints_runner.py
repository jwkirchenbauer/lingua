import os

OUTPUT_DIR = "/p/lustre5/kirchenb/common-pile-root/lingua/output/test_averaging/avg_checkpoint"

MODELS_TO_AVERAGE = [
    f"/p/lustre5/kirchenb/common-pile-root/lingua/output/test_averaging/checkpoints_hf/{ckpt}"
    for ckpt in [
        "0000230000",
        "0000231000",
        "0000232000",
    ]
]

print(f"Output directory: {OUTPUT_DIR}")
print(f"Models to average: {MODELS_TO_AVERAGE}")

cmd = f"""
python /p/lustre5/kirchenb/common-pile-root/lingua/average_hf_checkpoints.py \
--output_dir {OUTPUT_DIR} \
--model_paths {' '.join(MODELS_TO_AVERAGE)}
"""

print(f"Running command: {cmd}")
os.system(cmd)
print("Done!")
