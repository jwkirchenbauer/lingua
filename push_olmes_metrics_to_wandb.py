from lightning.pytorch.loggers import WandbLogger
import os
import json

for RUN_NAME, STEPS in [
    # ["prod_lingua_7B_curric_64N_phase1", [42_000]],
    # ["prod_lingua_7B_curric_64N_phase2", [84_000]],
    # ["prod_lingua_7B_wsd_128N", [80_000, 125_000]],
    # ["prod_lingua_7B_wsd_128N_orig", [125_000]],
    ["prod_lingua_64N", [40_000]],
]:
    for step_num in STEPS:
        STEP_10D = f"{step_num:010d}"
        SAVE_DIR = f"/p/lustre5/kirchenb/common-pile-root/lingua/output/{RUN_NAME}"
        EXP_VERSION = "v3"
        WANDB_TAGS = ["prod", "lingua", "olmes", EXP_VERSION]

        OLMES_SUBDIR = "olmes_evals"
        EVAL_METRICS_DIR = f"{SAVE_DIR}/{OLMES_SUBDIR}"
        SUMMARY_FILENAME = "metrics.json"
        METRICS_FILEPATH = f"{EVAL_METRICS_DIR}/{STEP_10D}/{SUMMARY_FILENAME}"

        with open(METRICS_FILEPATH, "r") as f:
            metrics_list = json.load(f)["all_primary_scores"]
        metrics_list = [
            s.replace(":rc::olmes:", "")
            .replace("::olmo1:", "")
            .replace(":rc::comma:", "")
            .replace("::starcoder_pass@10:", "")
            for s in metrics_list
        ]
        results = {s.split()[0]: float(s.split()[-1]) for s in metrics_list}

        # add any extra metrics here

        run_name_step_num = f"{RUN_NAME}_{step_num}"

        results["run_name_step_num"] = run_name_step_num
        if "phase" in RUN_NAME:
            results["run_name"] = RUN_NAME.split("_phase")[0]
        else:
            results["run_name"] = RUN_NAME
        results["step"] = step_num
        results["step_10d"] = STEP_10D

        print(json.dumps(results, indent=4))

        logger = WandbLogger(
            entity="tomg-group-umd",
            project="common",
            # name=results["run_name"],
            name=results["run_name_step_num"],
            save_dir=SAVE_DIR,
            tags=WANDB_TAGS,
        )
        logger.experiment.log(results, step=results["step"])

        # sync the wandb run to the server
        logger.experiment.finish()
