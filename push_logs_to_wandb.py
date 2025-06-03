from lightning.pytorch.loggers import WandbLogger
import os
import json

for RUN_NAME in [
    # "prod_lingua_7B_wsd_128N",
    # "prod_lingua_7B_curric_64N",
    # "prod_lingua_64N",
    # "prod_lingua_7B_wsd_128N",
    # "prod_lingua_7B_2T_128N",
    # "prod_lingua_7B_2T_long_hq_cd_128N",
    "prod_lingua_7B_2T_lin_hq_cd_128N",
]:
    SAVE_DIR = f"/p/lustre5/kirchenb/common-pile-root/lingua/output/{RUN_NAME}"
    EXP_VERSION = "v3"
    WANDB_TAGS = ["prod", "lingua", EXP_VERSION]

    METRICS_FILENAME = "metrics.jsonl"
    # METRICS_EVAL_FILENAME = "metrics.eval.jsonl"

    def col_rename_fn(row):
        # rename the keys in the row {from:to}
        rename_map = {
            "global_step": "optimizer_step",
            "loss/out": "global_loss",
            "optim/lr": "lr",
            "optim/total_tokens": "total_tokens",
            "optim/grad_norm": "global_grad_norm",
            "speed/curr_iter_time": "seconds_per_step",
        }
        new_row_critical = {}
        new_row_extra = {}
        for k, v in rename_map.items():
            if k in row:
                new_row_critical[v] = row[k]

        for k, v in row.items():
            if k not in rename_map:
                new_row_extra[k] = v
        new_row = dict()
        new_row.update({"step": new_row_critical["optimizer_step"]})
        new_row.update(new_row_critical)
        new_row.update(new_row_extra)

        return new_row

    with open(os.path.join(SAVE_DIR, METRICS_FILENAME), "r") as f:
        results = f.readlines()
    results = [json.loads(line) for line in results]

    results = [col_rename_fn(row) for row in results]
    print(f"Loaded {len(results)} results from {METRICS_FILENAME}")
    print(f"Results examples:{json.dumps(results[:5], indent=4)}")

    # now because there may have been restarts, we need to sort the results by step
    # and then take the latest instance of each step
    print(f"Before removing duplicates, we have {len(results)} results")
    results = sorted(results, key=lambda x: x["step"])
    results = {row["step"]: row for row in results}
    results = list(results.values())
    print(f"After removing duplicates, we have {len(results)} results")

    logger = WandbLogger(
        entity="tomg-group-umd",
        project="common",
        name=RUN_NAME,
        save_dir=SAVE_DIR,
        tags=WANDB_TAGS,
    )

    for result in results:
        logger.experiment.log(result, step=result["step"])

    # sync the wandb run to the server
    logger.experiment.finish()
