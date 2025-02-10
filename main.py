#!/usr/bin/env python3
# main.py

import os
import yaml

# dataset
from dataset.verifiers_dataset_loader import load_prompts_and_verifiers
from dataset.prompt_handler import prepare_prompts

# training
from train.grpo.grpo_model_loader import load_models
from train.grpo.train_with_block_sync import train_grpo_block_sync
from verifiers.combined_reward import combined_calculate_reward

# adapters (with lazy imports)
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
    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    elif args.config:
        raise FileNotFoundError(f"Could not find config file at path: {args.config}")

    cfg_base_model   = get_from_config(config, ["models", "base_model"], default=None)
    cfg_device       = get_from_config(config, ["models", "device"], default=None)
    cfg_load_adapter = get_from_config(config, ["training", "load_adapter_path"], default=None)
    cfg_save_adapter = get_from_config(config, ["training", "save_adapter_path"], default=None)
    cfg_verbose      = get_from_config(config, ["training", "verbose"], default=False)

    merged = {}
    merged["model"]            = override(args.model, cfg_base_model, default="Qwen/Qwen2.5-3B")
    merged["device"]           = override(args.device, cfg_device, default="auto")
    merged["load_adapter"]     = override(args.load_adapter_path, cfg_load_adapter, default=None)
    merged["save_adapter"]     = override(args.save_adapter_path, cfg_save_adapter, default=None)
    merged["verbose"]          = args.verbose or cfg_verbose
    merged["stages"]           = config.get("stages", [])

    merged["raw_config"] = config  # store entire config if needed
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
    """Your integrated reward function."""
    return combined_calculate_reward(response_text, item)


def train_single_stage(stage, base_model, ref_model, tokenizer, device, verbose):
    """
    Always use 'train_grpo_block_sync', ensuring the reference model is 
    frozen each block and then updated on schedule.
    """

    # 1) Extract dataset info
    dataset_conf = stage.get("dataset", {})
    dataset_path = dataset_conf.get("path", "dataset/zero/verifier_samples_very_easy.jsonl")
    prompt_template = dataset_conf.get("template", "src/cli/train/templates/prompt_template.jinja2")

    # 2) Coerce hyperparams
    lr     = float(stage.get("lr", 1e-5))
    epochs = int(float(stage.get("epochs", 5)))
    bsz    = int(float(stage.get("batch_size", 2)))
    G      = int(float(stage.get("G", 4)))

    # Additional block-sync / checkpoint params
    sync_every_n_batches      = stage.get("sync_every_n_batches", 50)  # default to 50 if not specified
    checkpoint_dir            = stage.get("checkpoint_dir", None)
    checkpoint_every_n_batches = stage.get("checkpoint_every_n_batches", None)
    checkpoint_every_n_epochs  = stage.get("checkpoint_every_n_epochs", None)

    # 3) Log stage info
    stage_name = stage.get("name", "no_name")
    method     = stage.get("method", "grpo")
    logger.info(color_text(f"==== Starting Stage: {stage_name} (method={method}) ====", YELLOW))
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Prompt template: {prompt_template}")
    logger.info(f"Hyperparams => LR={lr}, Epochs={epochs}, BatchSize={bsz}, G={G}, device={device}")
    logger.info(f"sync_every_n_batches={sync_every_n_batches}, checkpoint_dir={checkpoint_dir}")

    # 4) Load + prepare dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    from dataset.verifiers_dataset_loader import load_prompts_and_verifiers
    dataset_raw = load_prompts_and_verifiers(dataset_path)

    from dataset.prompt_handler import prepare_prompts
    prepared_dataset = prepare_prompts(dataset_raw, prompt_template)

    # 5) Always do block-sync training
    from train.grpo.train_with_block_sync import train_grpo_block_sync

    mean_loss, mean_reward = train_grpo_block_sync(
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=prepared_dataset,
        calculate_reward=integrated_reward,
        lr=lr,
        total_epochs=epochs,
        batch_size=bsz,
        G=G,
        device=device,
        verbose=verbose,
        kl_coeff=0.1,  # or stage.get("kl_coeff", 0.1)
        sync_every_n_batches=sync_every_n_batches,
        shuffle=True,
        # checkpointing
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n_batches=checkpoint_every_n_batches,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs
    )

    # 6) Log completion
    logger.info(color_text(
        f"Stage '{stage_name}' complete. Mean loss={mean_loss:.4f}, Reward={mean_reward:.4f}",
        GREEN
    ))

    return mean_loss, mean_reward


def main():
    # 1) Parse CLI arguments
    args = parse_arguments()

    # 2) Load + merge config
    merged = load_and_merge_config(args)

    # 3) Log final settings
    logger.info(f"Resolved Model: {merged['model']}")
    logger.info(f"Resolved Device: {merged['device'] or 'Auto-Detect'}")
    logger.info(f"Resolved load_adapter_path: {merged['load_adapter']}")
    logger.info(f"Resolved save_adapter_path: {merged['save_adapter']}")
    logger.info(f"Verbose: {merged['verbose']}")

    # 4) Load the training models
    from train.grpo.grpo_model_loader import load_models
    base_model, ref_model, tokenizer, device = load_models(merged["model"], merged["device"])

    # 5) Load adapters if user specified path
    if merged["load_adapter"] is not None:
        path = merged["load_adapter"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find adapter file: {path}")
        logger.info(f"Loading adapters from {path}...")
        load_adapters(base_model, path)

    # 6) Retrieve or define stages
    raw_stages = merged["stages"]
    if not raw_stages:
        raw_stages = [{
            "name": "stage1",
            "method": "grpo",
            "lr": 1e-6,
            "epochs": 5,
            "batch_size": 2,
            "G": 4
        }]
    stages = raw_stages

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
