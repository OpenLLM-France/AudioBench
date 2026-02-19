import fire
import yaml
import logging
import torch
from tqdm import tqdm
from src.main_evaluate import run_evaluation

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate_models_from_config(config_path="configs/config.yaml"):
    config = load_config(config_path)

    global_params = config.get("global", {})
    global_datasets = global_params.get("datasets", [])

    models = config.get("models", [])
    pbar = tqdm(models, desc="Processing models")

    logger.info("= = "*20)
    logger.info(f"Running : {config_path}")
    logger.info(f"Datasets ({len(global_datasets)}): {global_datasets}")
    logger.info(f"Models ({len(models)}): {models}")
    logger.info(f"Number of samples: {global_params.get('number_of_samples', -1)}")
    logger.info("= = "*20)

    global_backend = global_params.get("backend", "transformers")

    for model_config in pbar:
        model_name = model_config["name"]
        pbar.set_description(f"Processing model: {model_name}")

        model_batch_size = model_config.get("batch_size", global_params.get("batch_size", 1))
        model_backend = model_config.get("backend", global_backend)

        model_datasets = model_config.get("datasets", global_datasets)

        model = None

        for dataset_config in model_datasets:
            dataset_name = dataset_config["name"]
            metrics = dataset_config.get("metrics", None)

            model = run_evaluation(
                dataset_name=dataset_name,
                model_name=model_name,
                batch_size=model_batch_size,
                overwrite=global_params.get("overwrite", False),
                metrics=metrics,
                number_of_samples=global_params.get("number_of_samples", -1),
                log_folder=global_params.get("output_folder", "results"),
                backend=model_backend,
                model=model
            )

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    fire.Fire(evaluate_models_from_config)
