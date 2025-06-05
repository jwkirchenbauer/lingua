from huggingface_hub import HfApi, HfFolder, Repository
import os


TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

MODEL_PATH = "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_lin_hq_cd_128N/avg_checkpoint"

REPO_ID = "common-pile/comma-v0.1-2t"

README_PATH = (
    "/p/lustre5/kirchenb/common-pile-root/lingua/hf_model_readmes/comma-v0-1-2t.md"
)

api = HfApi(token=TOKEN)

api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=False,
    private=False,
)

api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=REPO_ID,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj=README_PATH,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)
