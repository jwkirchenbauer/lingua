import os
from datasets import load_dataset, get_dataset_config_names

cpu_count = os.cpu_count()

DS_REPO_NAME = "nkandpa2/common-pile-chunked"

# space must be large enough to download all subsets
DS_CACHE_DIR = "/p/vast1/pretrain/datasets/common_pile/.cache"

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
    while retry_count < RETRY_LIMIT:
        try:
            ds = load_dataset(
                DS_REPO_NAME,
                ds_config,
                cache_dir=DS_CACHE_DIR,
                num_proc=cpu_count,
            )
            break
        except Exception as e:
            print(f"(Try {retry_count}) Failed to load {ds_config}: {e}")
            retry_count += 1
            if retry_count >= RETRY_LIMIT:
                print(f"Exceeded retry limit for {ds_config}.")
                break
    print(f"Loaded {ds_config} successfully.")
    print(ds)
