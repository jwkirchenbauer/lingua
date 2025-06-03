# Created primarily by Gemini 2.5 Flash with some modifications.

import os
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict
import argparse
import json
import shutil


def average_model_weights(model_checkpoint_paths, output_dir):
    """
    Averages the weights of multiple Hugging Face model checkpoints on a per-parameter
    basis, handling sharded safetensors files efficiently.

    Args:
        model_checkpoint_paths (list[str]): A list of paths to the Hugging Face model checkpoints.
        output_dir (str): The directory to save the averaged model.
    """
    if not model_checkpoint_paths:
        raise ValueError("model_checkpoint_paths cannot be empty.")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Averaging weights for {len(model_checkpoint_paths)} models.")

    # Load the config from the first model to save the averaged model with correct config
    config = AutoConfig.from_pretrained(model_checkpoint_paths[0])

    # Check for tie_word_embeddings setting
    # Default to False: If tie_word_embeddings is not explicitly in the config,
    # we assume it's NOT tied, and thus lm_head.weight should be present.
    # If lm_head.weight is then missing, it will assert and fail.
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
    print(f"Model config 'tie_word_embeddings': {tie_word_embeddings}")

    # Initialize a model with the config to get ALL expected parameter names from its state_dict
    reference_model_for_keys = AutoModelForCausalLM.from_config(config)
    all_param_names = list(reference_model_for_keys.state_dict().keys())
    del reference_model_for_keys  # Free up memory

    # Dictionary to store file paths for each parameter across all models
    param_files_map = defaultdict(lambda: defaultdict(list))

    # First pass: Build a map of parameter names to their sharded file paths
    # and the corresponding key within that safetensors file
    print("Mapping parameter locations across checkpoints...")
    for model_path in model_checkpoint_paths:
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        model_state_dict_keys_in_checkpoint = set()

        if os.path.exists(index_file):
            # Load the index file to find sharding information
            with open(index_file, "r") as f:
                index_data = json.load(f)

            for param_name, relative_file_path in index_data["weight_map"].items():
                full_file_path = os.path.join(model_path, relative_file_path)
                param_files_map[param_name][model_path].append(
                    {"file": full_file_path, "key": param_name}
                )
                model_state_dict_keys_in_checkpoint.add(param_name)
        else:
            # Assume a single safetensors file
            single_sf_file = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(single_sf_file):
                raise FileNotFoundError(
                    f"No safetensors files found in {model_path}. Expected model.safetensors or sharded files."
                )
            with safe_open(single_sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    param_files_map[key][model_path].append(
                        {"file": single_sf_file, "key": key}
                    )
                    model_state_dict_keys_in_checkpoint.add(key)

    averaged_state_dict = {}

    # Iterate through each parameter and average its weights
    print("Averaging parameters (this may take a while for large models)...")
    for i, param_name in enumerate(all_param_names):
        if i % 10 == 0:
            print(f"Processing parameter {i}/{len(all_param_names)}: {param_name}")

        param_tensors = []

        # Collect the specific parameter from all models
        for model_path in model_checkpoint_paths:
            file_info_list = param_files_map[param_name][model_path]

            if not file_info_list:
                # This parameter is not explicitly saved in this model's safetensors files.
                # This typically happens for tied weights (like lm_head.weight being tied to transformer.wte.weight).
                if param_name == "lm_head.weight":
                    # If lm_head.weight is missing, assert tie_word_embeddings is True
                    # If tie_word_embeddings is False (or not present in config, thus defaulting to False),
                    # and lm_head.weight is missing, this will trigger an AssertionError.
                    assert tie_word_embeddings, (
                        f"lm_head.weight is missing from {model_path} but tie_word_embeddings is False/unset in config. "
                        "If lm_head.weight is meant to be tied to embeddings, ensure 'tie_word_embeddings: true' "
                        "is explicitly set in the model's config.json. Otherwise, it indicates a corrupted checkpoint."
                    )

                    # Assume it's tied to transformer.wte.weight
                    wte_file_info_list = param_files_map["transformer.wte.weight"][
                        model_path
                    ]
                    if wte_file_info_list:
                        wte_file_info = wte_file_info_list[0]
                        with safe_open(
                            wte_file_info["file"], framework="pt", device="cpu"
                        ) as f:
                            tied_tensor = f.get_tensor(wte_file_info["key"])
                            param_tensors.append(tied_tensor)
                    else:
                        raise RuntimeError(
                            f"Cannot find 'transformer.wte.weight' for tied 'lm_head.weight' in {model_path}"
                        )
                else:
                    # If it's not lm_head.weight or another specific known tied weight,
                    # this indicates a more general missing parameter.
                    raise RuntimeError(
                        f"Parameter '{param_name}' not found in checkpoint files for {model_path}."
                    )
            else:
                # Regular parameter, load it as usual
                file_info = file_info_list[
                    0
                ]  # There should be only one entry per param per model
                with safe_open(file_info["file"], framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(file_info["key"])
                    param_tensors.append(tensor)

        # Perform the averaging
        if param_tensors:
            averaged_tensor = torch.stack(param_tensors).mean(dim=0)
            averaged_state_dict[param_name] = averaged_tensor

        # Clear param_tensors to free memory for the next parameter
        del param_tensors

    print("Saving averaged model...")

    # Create a dummy model to load the state dict into
    model_to_save = AutoModelForCausalLM.from_config(config)

    # Check for missing keys before loading to provide more specific errors
    missing_keys_before_load = set(all_param_names) - set(averaged_state_dict.keys())
    if missing_keys_before_load:
        print(
            f"Warning: The following expected parameters are missing from averaged_state_dict: {missing_keys_before_load}"
        )
        # This warning might be okay if 'all_param_names' contains keys that are not directly
        # saved in the safetensors but are dynamically generated/tied during model initialization
        # (e.g., in some obscure cases beyond lm_head.weight).
        # The strict=True in load_state_dict will catch critical ones.

    model_to_save.load_state_dict(averaged_state_dict, strict=True)
    model_to_save.save_pretrained(output_dir)

    print(f"Averaged model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Average weights of multiple Hugging Face model checkpoints."
    )
    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=False,  # It should be required for actual use
        help="A list of local Hugging Face model directory paths (e.g., ./model_dir1 ./model_dir2).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./averaged_model",
        required=False,
        help="The directory to save the averaged model. Defaults to './averaged_model'.",
    )
    args = parser.parse_args()

    if args.model_paths:

        print(f"Using provided model paths: {args.model_paths}")
        output_directory = args.output_dir
        model_paths_to_use = args.model_paths
        average_model_weights(model_paths_to_use, output_directory)

        # copy any file with *tok*.json in the first model directory to the output directory
        print("Copying tokenizer files from the first model directory to the output directory...")
        first_model_path = model_paths_to_use[0]
        tok_files = [
            f for f in os.listdir(first_model_path) if "tok" in f and f.endswith(".json")
        ]
        if tok_files:
            for tok_file in tok_files:
                src_path = os.path.join(first_model_path, tok_file)
                dest_path = os.path.join(output_directory, tok_file)
                shutil.copy(src_path, dest_path)
                print(f"Copied {tok_file} to {output_directory}")
        exit(0)

    print("Creating dummy models for demonstration...")
    dummy_model_paths = ["./dummy_model_1", "./dummy_model_2", "./dummy_model_3"]

    # Clean up any previous runs
    if os.path.exists("./model_avg_output"):
        shutil.rmtree("./model_avg_output")
    for p in dummy_model_paths:
        if os.path.exists(p):
            shutil.rmtree(p)

    # Create dummy model 1
    model1 = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    with torch.no_grad():
        model1.transformer.h[0].attn.c_attn.weight.data += 0.1
    model1.save_pretrained("./dummy_model_1")

    # Create dummy model 2
    model2 = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    with torch.no_grad():
        model2.transformer.h[0].attn.c_attn.weight.data -= 0.1
    model2.save_pretrained("./dummy_model_2")

    # Create dummy model 3
    model3 = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    model3.save_pretrained("./dummy_model_3")
    print("Dummy models created.")

    output_directory = args.output_dir
    model_paths_to_use = dummy_model_paths
    average_model_weights(model_paths_to_use, output_directory)

    print("\nVerification: Loading averaged model and inspecting a parameter.")
    averaged_model = AutoModelForCausalLM.from_pretrained(output_directory)

    param_name_to_check = "transformer.h.0.attn.c_attn.weight"
    lm_head_name_to_check = "lm_head.weight"

    param_tensors_for_verification = []
    lm_head_tensors_for_verification = []

    for model_path in model_paths_to_use:
        sf_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(sf_file):
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                if param_name_to_check in f.keys():
                    param_tensors_for_verification.append(
                        f.get_tensor(param_name_to_check)
                    )
                if lm_head_name_to_check in f.keys():
                    lm_head_tensors_for_verification.append(
                        f.get_tensor(lm_head_name_to_check)
                    )
                elif "transformer.wte.weight" in f.keys():  # Check for tied weight
                    lm_head_tensors_for_verification.append(
                        f.get_tensor("transformer.wte.weight")
                    )
                else:
                    print(
                        f"Warning: Neither '{lm_head_name_to_check}' nor 'transformer.wte.weight' found in {sf_file} for verification."
                    )
        else:  # Try index.json for sharded models
            index_file = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                with open(index_file, "r") as f:
                    index_data = json.load(f)

                # Check for regular parameter
                if param_name_to_check in index_data["weight_map"]:
                    relative_path = index_data["weight_map"][param_name_to_check]
                    full_path = os.path.join(model_path, relative_path)
                    with safe_open(full_path, framework="pt", device="cpu") as f_shard:
                        param_tensors_for_verification.append(
                            f_shard.get_tensor(param_name_to_check)
                        )

                # Check for lm_head.weight (could be tied)
                if lm_head_name_to_check in index_data["weight_map"]:
                    relative_path = index_data["weight_map"][lm_head_name_to_check]
                    full_path = os.path.join(model_path, relative_path)
                    with safe_open(full_path, framework="pt", device="cpu") as f_shard:
                        lm_head_tensors_for_verification.append(
                            f_shard.get_tensor(lm_head_name_to_check)
                        )
                elif "transformer.wte.weight" in index_data["weight_map"]:
                    relative_path = index_data["weight_map"]["transformer.wte.weight"]
                    full_path = os.path.join(model_path, relative_path)
                    with safe_open(full_path, framework="pt", device="cpu") as f_shard:
                        lm_head_tensors_for_verification.append(
                            f_shard.get_tensor("transformer.wte.weight")
                        )
                else:
                    print(
                        f"Warning: Neither '{lm_head_name_to_check}' nor 'transformer.wte.weight' found in index for {model_path} for verification."
                    )
            else:
                print(
                    f"Warning: Neither model.safetensors nor model.safetensors.index.json found in {model_path} for verification."
                )

    if (
        len(param_tensors_for_verification) == len(model_paths_to_use)
        and len(param_tensors_for_verification) > 0
    ):
        avg_calculated = torch.stack(param_tensors_for_verification).mean(dim=0)
        avg_loaded = averaged_model.state_dict()[param_name_to_check]
        is_close = torch.allclose(avg_calculated, avg_loaded, atol=1e-5)
        print(
            f"Are calculated average and loaded average for '{param_name_to_check}' close? {is_close}"
        )
        if not is_close:
            print(
                "Mismatch detected! Max difference:",
                torch.max(torch.abs(avg_calculated - avg_loaded)).item(),
            )
    else:
        print(
            f"Skipping detailed verification for '{param_name_to_check}' due to missing parameters in some models."
        )

    if (
        len(lm_head_tensors_for_verification) == len(model_paths_to_use)
        and len(lm_head_tensors_for_verification) > 0
    ):
        avg_calculated_lm_head = torch.stack(lm_head_tensors_for_verification).mean(
            dim=0
        )
        avg_loaded_lm_head = averaged_model.state_dict()[lm_head_name_to_check]
        is_close_lm_head = torch.allclose(
            avg_calculated_lm_head, avg_loaded_lm_head, atol=1e-5
        )
        print(
            f"Are calculated average and loaded average for '{lm_head_name_to_check}' close? {is_close_lm_head}"
        )
        if not is_close_lm_head:
            print(
                "Mismatch detected for lm_head.weight! Max difference:",
                torch.max(
                    torch.abs(avg_calculated_lm_head - avg_loaded_lm_head)
                ).item(),
            )
    else:
        print(
            f"Skipping detailed verification for '{lm_head_name_to_check}' due to missing parameters in some models."
        )

    # Clean up dummy models
    print("Cleaning up dummy models...")
    for p in dummy_model_paths:
        if os.path.exists(p):
            shutil.rmtree(p)
    # Keep output_directory for inspection if needed, or uncomment to remove
    # if os.path.exists(output_directory):
    #     shutil.rmtree(output_directory)
