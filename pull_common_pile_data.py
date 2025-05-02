import os
from datasets import load_dataset, get_dataset_config_names
import glob

cpu_count = os.cpu_count()

DS_REPO_NAME = "nkandpa2/common-pile-chunked"

# space must be large enough to download all subsets
DS_CACHE_DIR = "/p/vast1/pretrain/datasets/common_pile/.cache"

DS_RAW_JSONS_DIR = "/p/vast1/pretrain/datasets/common_pile/common-pile-chunked"

DS_SAVE_DIR = "/p/vast1/pretrain/datasets/common_pile/hf_format"

# debugging
# all_config_names = [
#     "pressbooks",
#     "public_domain_review",
# ]

# this will include the default subset which is redundant/the union
all_config_names = get_dataset_config_names(DS_REPO_NAME)

# reorder
all_config_names = [n for n in all_config_names if n != "default"] + ["default"]

print(all_config_names)
# exit()

RETRY_LIMIT = 10
for ds_config in all_config_names:
    print(f"Loading {ds_config}...")
    retry_count = 0
    ds = None
    if os.path.isdir(f"{DS_SAVE_DIR}/{ds_config}"):
        print(f"Already exists: {DS_SAVE_DIR}/{ds_config}")
        continue
    while retry_count < RETRY_LIMIT:
        try:
            # ds = load_dataset(
            #     DS_REPO_NAME,
            #     ds_config,
            #     cache_dir=DS_CACHE_DIR,
            #     num_proc=cpu_count,
            # )
            all_matching_files = sorted(
                glob.glob(
                    f"{DS_RAW_JSONS_DIR}/{ds_config}/{ds_config}.chunk.*.jsonl",
                    recursive=True,
                )
            )
            print(all_matching_files)
            ds = load_dataset(
                "json",
                data_files=all_matching_files,
                cache_dir=DS_CACHE_DIR,
                num_proc=cpu_count,
            )
            print(ds)
            break
        except Exception as e:
            print(f"(Try {retry_count}) Failed to load {ds_config}: {e}")
            retry_count += 1
            if retry_count >= RETRY_LIMIT:
                print(f"Exceeded retry limit for {ds_config}.")
                break
    print(f"Loaded {ds_config} successfully.")
    print(ds)
    ds.save_to_disk(f"{DS_SAVE_DIR}/{ds_config}", num_proc=cpu_count)
