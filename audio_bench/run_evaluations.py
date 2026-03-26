import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import fire
import yaml
import gc
import logging
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from audio_bench.main_evaluate import run_evaluation

logger = logging.getLogger(__name__)

def _flatten_datasets(dataset_list):
    """Expand group entries into individual dataset entries with inherited properties.

    Supports nested groups: a child that itself has a 'group' key is recursively
    expanded.  Property inheritance cascades — parent props merge into child,
    child overrides win.
    """
    result = []
    for entry in dataset_list:
        if "group" in entry:
            group_props = {k: v for k, v in entry.items() if k not in ("group", "datasets")}
            children = []
            for child in entry.get("datasets", []):
                children.append({**group_props, **child})
            result.extend(_flatten_datasets(children))
        else:
            result.append(entry)
    return result

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate_models_from_config(config_path="configs/test.yaml"):
    config = load_config(config_path)

    global_params = config.get("global", {})
    global_datasets = _flatten_datasets(global_params.get("datasets", []))

    models = config.get("models", [])
    logger.info(" ="*30)
    logger.info(f"Running benchmark using : {config_path}")
    logger.info(f"Datasets ({len(global_datasets)}): {global_datasets}")
    logger.info(f"Models ({len(models)}): {models}")
    logger.info(f"Number of samples: {global_params.get('number_of_samples', -1)}")
    logger.info(" ="*30)
    logger.info("\n"*3)

    global_backend = global_params.get("backend", "transformers")

    with logging_redirect_tqdm():
        pbar = tqdm(models, desc="Processing models", leave=False)

        for model_config in pbar:
            model_name = model_config["name"]
            pbar.set_description(f"Processing model: {model_name}")

            model_batch_size = model_config.get("batch_size", global_params.get("batch_size", 1))
            model_backend = model_config.get("backend", global_backend)
            model_device = model_config.get("device", global_params.get("device"))

            model_datasets = _flatten_datasets(model_config.get("datasets", [])) if "datasets" in model_config else global_datasets

            model = None

            # PASS 1: Inference
            try:
                dataset_pbar = tqdm(model_datasets, desc=model_name, leave=False)
                for dataset_config in dataset_pbar:
                    dataset_name = dataset_config["name"]
                    dataset_pbar.set_description(f"{model_name} | {dataset_name}")

                    dataset_config['number_of_samples'] = dataset_config.get("number_of_samples", global_params.get("number_of_samples", -1))
                    dataset_config['min_audio_duration'] = dataset_config.get("min_audio_duration", global_params.get("min_audio_duration"))
                    dataset_config['max_audio_duration'] = dataset_config.get("max_audio_duration", global_params.get("max_audio_duration"))
                    dataset_config['ignore_offsets'] = dataset_config.get("ignore_offsets", global_params.get("ignore_offsets", False))

                    evaluation_model_config = dict(
                        batch_size=dataset_config.get("batch_size", model_batch_size),
                        backend=model_backend,
                        path=model_config.get("path"),
                        gpu_memory_utilization=model_config.get("gpu_memory_utilization", global_params.get("gpu_memory_utilization", 0.4)),
                        device=model_device,
                    )

                    try:
                        model = run_evaluation(
                            dataset_name=dataset_name,
                            dataset_config=dataset_config,
                            model_name=model_name,
                            model_config=evaluation_model_config,
                            model=model,
                            overwrite=global_params.get("overwrite", False),
                            log_folder=global_params.get("output_folder", "results"),
                            compute_metrics=global_params.get("compute_metrics", True),
                            skip_inference=global_params.get("skip_inference", False),
                        )
                    except Exception as e:
                        if global_params.get("skip_errors", False):
                            logger.error(f"Error evaluating model {model_name} on dataset {dataset_name}: {e}")
                        else:
                            raise Exception(f"Error evaluating model {model_name} on dataset {dataset_name}: {e}") from e
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                dataset_pbar.close()
                if model is not None:
                    if hasattr(model, 'destroy'):
                        model.destroy()
                    del model
                gc.collect()
                torch.cuda.empty_cache()

if __name__ == "__main__":
    fire.Fire(evaluate_models_from_config)
