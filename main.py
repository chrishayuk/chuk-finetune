#!/usr/bin/env python3
# main.py
import os
import yaml

# dataset
from dataset.verifiers_dataset_loader import load_prompts_and_verifiers
from dataset.prompt_handler import prepare_prompts

# training
from train.grpo.grpo_model_loader import load_models
from train.grpo.grpo_trainer import train_grpo
from verifiers.combined_reward import combined_calculate_reward

# adapters
from model.adapters import save_adapters, load_adapters

# CLI imports
from cli.train.arg_parser import parse_arguments
from cli.train.logger_config import YELLOW, logger, color_text, BOLD, GREEN
from cli.train.training_monitor import monitor_training_progress


def get_from_config(cfg, keys, default=None):
    """
    Safely retrieve nested keys from a dict, or return 'default' if not found.
    Example usage: get_from_config(config, ["models", "base_model"], default="foo").
    """
    current = cfg
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def override(cli_val, config_val, default=None):
    """
    Merge command-line (CLI) values with config values.
    Priority order:
      1) CLI arg if provided
      2) Config value if provided
      3) Otherwise the default
    """
    return cli_val if cli_val is not None else (config_val if config_val is not None else default)


def load_and_merge_config(args):
    """
    Loads YAML config if specified, merges with CLI args, and returns final resolved settings.
    Returns:
        A dictionary with the merged config plus top-level fields for model/device etc.
    """
    # 1. Load YAML config if provided
    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    elif args.config:
        raise FileNotFoundError(f"Could not find config file at path: {args.config}")

    # 2. Extract config fields
    cfg_base_model   = get_from_config(config, ["models", "base_model"], default=None)
    cfg_device       = get_from_config(config, ["models", "device"], default=None)
    cfg_load_adapter = get_from_config(config, ["training", "load_adapter_path"], default=None)
    cfg_save_adapter = get_from_config(config, ["training", "save_adapter_path"], default=None)
    cfg_verbose      = get_from_config(config, ["training", "verbose"], default=False)

    # 3. Merge CLI overrides
    merged = {}
    merged["model"]            = override(args.model, cfg_base_model, default="Qwen/Qwen2.5-3B")
    merged["device"]           = override(args.device, cfg_device, default="auto")
    merged["load_adapter"]     = override(args.load_adapter_path, cfg_load_adapter, default=None)
    merged["save_adapter"]     = override(args.save_adapter_path, cfg_save_adapter, default=None)
    merged["verbose"]          = args.verbose or cfg_verbose
    merged["stages"]           = config.get("stages", [])

    # We'll keep the entire config in merged["raw_config"] if needed
    merged["raw_config"] = config

    return merged


def load_or_default_stages(raw_stages):
    """
    If no stages are defined in config, return a default single stage.
    Else, return whatever is in the config.
    """
    if not raw_stages:
        return [{
            "name": "stage1",
            "method": "grpo",
            "dataset": {
                "path": "dataset/zero/verifier_samples_very_easy.jsonl",
                "template": "src/cli/train/templates/prompt_template.jinja2"
            },
            "lr": 1e-5,
            "epochs": 5,
            "batch_size": 2,
            "G": 4
        }]
    return raw_stages


def integrated_reward(response_text, item):
    """Define or import your integrated reward function."""
    return combined_calculate_reward(response_text, item)


def train_single_stage(stage, base_model, ref_model, tokenizer, device, verbose):
    """
    Train a single stage using the given parameters (GRPO or another method).
    Returns any training stats (mean_loss, mean_reward, etc.) if desired.
    """

    # 1. Extract dataset info
    dataset_conf = stage.get("dataset", {})
    dataset_path = dataset_conf.get("path", "dataset/zero/verifier_samples_very_easy.jsonl")
    prompt_template = dataset_conf.get("template", "src/cli/train/templates/prompt_template.jinja2")

    # 2. Coerce hyperparams
    lr     = float(stage.get("lr", 1e-5))
    epochs = int(float(stage.get("epochs", 5)))
    bsz    = int(float(stage.get("batch_size", 2)))
    G      = int(float(stage.get("G", 4)))

    # 3. Log stage info
    stage_name = stage.get("name", "no_name")
    method     = stage.get("method", "grpo")
    logger.info(color_text(f"==== Starting Stage: {stage_name} (method={method}) ====", YELLOW))
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Prompt template: {prompt_template}")
    logger.info(f"Hyperparams => LR={lr}, Epochs={epochs}, BatchSize={bsz}, G={G}")

    # 4. Load and prepare the dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_prompts_and_verifiers(dataset_path)

    logger.info(f"Preparing prompts using template: {prompt_template}")
    prepared_dataset = prepare_prompts(dataset, prompt_template)

    # 5. Train the model
    gen = None
    if method == "grpo":
        gen = train_grpo(
            base_model=base_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=prepared_dataset,
            calculate_reward=integrated_reward,
            lr=lr,
            epochs=epochs,
            batch_size=bsz,
            G=G,
            device=device,
            verbose=verbose,
            as_generator=True
        )
    else:
        # If you support other methods, handle them here
        raise ValueError(f"Unsupported method: {method}")

    # 6. Monitor progress if your trainer returns generator events
    mean_loss, mean_reward = monitor_training_progress(gen)

    # 7. Log stage completion
    logger.info(color_text(
        f"Stage '{stage_name}' complete. Mean loss={mean_loss:.4f}, Reward={mean_reward:.4f}",
        GREEN
    ))

    # Return any stats if needed
    return mean_loss, mean_reward


def main():
    # 1) Parse CLI arguments
    args = parse_arguments()

    # 2) Load + merge config and CLI
    merged = load_and_merge_config(args)

    # 3) Log final settings
    logger.info(f"Resolved Model: {merged['model']}")
    logger.info(f"Resolved Device: {merged['device'] or 'Auto-Detect'}")
    logger.info(f"Resolved load_adapter_path: {merged['load_adapter']}")
    logger.info(f"Resolved save_adapter_path: {merged['save_adapter']}")
    logger.info(f"Verbose: {merged['verbose']}")

    # 4) Load the training models
    base_model, ref_model, tokenizer, device = load_models(merged["model"], merged["device"])

    # 5) Load adapters if user specified a path
    if merged["load_adapter"] is not None:
        path = merged["load_adapter"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find adapter file: {path}")
        logger.info(f"Loading adapters from {path}...")
        load_adapters(base_model, path)

    # 6) Retrieve or define stages
    raw_stages = merged["stages"]
    stages = load_or_default_stages(raw_stages)

    # 7) Train each stage
    for stage in stages:
        train_single_stage(stage, base_model, ref_model, tokenizer, device, merged["verbose"])

    # 8) After all stages, optionally save adapters
    if merged["save_adapter"] is not None:
        path = merged["save_adapter"]
        logger.info(f"Saving adapters to {path}...")
        save_adapters(base_model, path)

    logger.info(color_text("===== All Training Stages Complete! =====", BOLD))


if __name__ == "__main__":
    main()