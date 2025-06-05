from huggingface_hub import HfApi, HfFolder, Repository
import os


TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

FOLDER_PATH = (
    "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_lingua_upload"
)

# TODO automate the copy of required files from the real sources into this dir, for now manually copied a few things

# src paths are absolute and point to the original files
# dst paths are relative to the FOLDER_PATH and point to the files as they should be in the repo
SRC_TO_DST = {
    "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_128N/config.yaml": "config.yaml",
    "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_128N/metrics.jsonl": "metrics.jsonl",
    "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_lin_hq_cd_128N/config.yaml": "config.hq_cd.yaml",
    "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_lin_hq_cd_128N/metrics.jsonl": "metrics.hq_cd.jsonl",
}

os.makedirs(FOLDER_PATH, exist_ok=True)

for src, dst in SRC_TO_DST.items():
    dst_path = os.path.join(FOLDER_PATH, dst)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    os.system(f"cp {src} {dst_path}")


REPO_ID = "common-pile/comma-v0.1-2t-checkpoints"

api = HfApi(token=TOKEN)

api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=False,
    private=False,
)

api.upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=REPO_ID,
    repo_type="model",
)
