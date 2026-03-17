
import fire
import json
import gc
import torch
import logging
from pathlib import Path
from tqdm import tqdm

from src.dataset_factory import load_dataset_processor
from model_factory import load_model

# =  =  =  =  =  =  =  =  =  =  =  Helpers  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

def _load_predictions(path):
    """Load a prediction file, handling both old (list) and new (dict with metadata) formats.
    Returns (predictions_list, metadata_dict).
    """
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw, {}
    return raw.get("predictions", raw), raw.get("metadata", {})

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


def do_model_prediction(input_data, model):
    batch_size = model.batch_size

    if model.backend=="vllm":
        # Process in batches to avoid massive RAM spike from building chat messages
        vllm_batch_size = max(100, batch_size*4)
        model_predictions = []
        for i in tqdm(range(0, len(input_data), vllm_batch_size), desc="vLLM Inference Batches", leave=False):
            batch = input_data[i:i + vllm_batch_size]
            results = model.generate(batch)
            model_predictions.extend(results)
        return model_predictions

    if batch_size <= 1:
        model_predictions = []
        for inputs in tqdm(input_data, leave=False):
            outputs = model.generate(inputs)
            if isinstance(outputs, list):
                model_predictions.extend(outputs)
            else:
                model_predictions.append(outputs)
        return model_predictions

    # batch_size > 1 : chunk and process per batch
    model_predictions = []
    num_batches = (len(input_data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(input_data), batch_size), total=num_batches, leave=False, desc=f"Batch inference (bs={batch_size})"):
        batch = input_data[i:i + batch_size]
        outputs = model.generate(batch)
        if isinstance(outputs, list):
            model_predictions.extend(outputs)
        else:
            model_predictions.append(outputs)
    return model_predictions

def run_evaluation(
        dataset_name: str = None,
        dataset_config: dict = None,
        model_name: str = None,
        model_config: dict = None,
        model = None,
        overwrite: bool = False,
        log_folder: str = "log_for_all_models",
        compute_metrics: bool = True,
        skip_inference: bool = False,
    ):
    
    processor = load_dataset_processor(
        dataset_name,
        number_of_samples=dataset_config.get("number_of_samples"),
        dataset_path=dataset_config.get("path"),
        min_audio_duration=dataset_config.get("min_audio_duration"),
        max_audio_duration=dataset_config.get("max_audio_duration"),
        ignore_offsets=dataset_config.get("ignore_offsets", False),
    )
    
    if dataset_config.get("task") is not None:
        processor.task_type = dataset_config.get("task").upper()
        
    if dataset_config.get("sub_task") is not None:
        processor.sub_task = dataset_config.get("sub_task")

    if dataset_config.get("language") is not None:
        processor.language = dataset_config.get("language").upper()
    
    if dataset_config.get("metrics") is None:
        if processor.metrics is not None:
            dataset_config["metrics"] = processor.metrics
            logger.info(f"Metrics is not specified. Use the default metrics of the dataset: {dataset_config.get('metrics')}")
        else:
            raise NotImplementedError(f"The dataset {dataset_name} does not have a default metrics.")

    if isinstance(dataset_config["metrics"], str):
        dataset_config["metrics"] = [dataset_config["metrics"]]

    if processor.language is not None and processor.language!="UNKNOWN":
        xp_dir = Path(log_folder) / model_name / processor.language
    else:
        xp_dir = xp_dir = Path(log_folder) / model_name
    score_path = xp_dir / f"{dataset_name}_score.json"
    prediction_path = xp_dir / f"{dataset_name}.json"

    xp_dir.mkdir(parents=True, exist_ok=True)

    if model_name == 'wavllm_fairseq':
        model_config["batch_size"] = -1
        if model is not None:
            model.batch_size = -1
        logger.info("Batch size is set to -1 for wavllm_fairseq model.")

    if not overwrite and prediction_path.exists():
        predictions, pred_metadata = _load_predictions(prediction_path)
        requested = dataset_config.get("number_of_samples", -1)
        if requested > 0 and len(predictions) < requested:
            prev_dataset_size = pred_metadata.get("dataset_size")
            if prev_dataset_size is None and score_path.exists():
                prev_dataset_size = json.loads(score_path.read_text()).get("dataset_size")
            if prev_dataset_size is None or len(predictions) < prev_dataset_size:
                overwrite = True
                logger.info(f"Found {len(predictions)} samples in {prediction_path} instead of {requested}. Overwrite set to True.")
    
    if not overwrite and score_path.exists():
        results = json.loads(score_path.read_text())
        if (all([metric in results for metric in dataset_config["metrics"]])):
            logger.info('- '*30)
            logger.info(f"Evaluation for {model_name.upper()} and {dataset_name.upper()} exists. Skip the evaluation.")
            logger.info(results)
            logger.info('- '*30)
            logger.info("\n"*3)
            return model

    if not skip_inference and (overwrite or not prediction_path.exists()):
        if overwrite:
             logger.info(f"Overwrite is enabled. Try to infer {dataset_name} with {model_name}.")
        else:
            logger.info(f"No results found for {dataset_name} with {model_name}.")

        # Load dataset (deferred until now to skip download when not needed)
        input_data = processor.load()

        # Load model
        if model is None:
            model = load_model(model_name, backend=model_config["backend"], model_path=model_config.get("path"), batch_size=model_config["batch_size"])

        # Specific current dataset name for evaluation
        model.dataset_name = dataset_name

        # Sync batch_size for per-dataset overrides (safe: vLLM models aren't reused across datasets)
        model.batch_size = model_config["batch_size"]

        # Infer with model
        model_predictions = do_model_prediction(input_data, model)
        data_with_model_predictions = processor.format_model_predictions(input_data, model_predictions)
        
        # Free memory associated with raw audio data
        del model_predictions
        input_data.clear() # If it's a list, clear it.
        del input_data
        # Save the result with predictions (wrapped with metadata)
        prediction_data = {
            "metadata": {
                "dataset_size": processor._dataset_size,
                "number_of_samples": len(data_with_model_predictions),
            },
            "predictions": data_with_model_predictions,
        }
        with open(prediction_path, 'w') as f:
            json.dump(prediction_data, f, indent=4, ensure_ascii=False)
        del data_with_model_predictions



    if not prediction_path.exists():
        logger.error(f"Prediction file {prediction_path} not found. Cannot compute metrics.")
        del processor
        return model
    
    if compute_metrics:
        data_with_model_predictions, pred_metadata = _load_predictions(prediction_path)
        results = dict()
        results['model_name'] = model_name
        results['dataset_name'] = dataset_name
        results['metrics'] = dataset_config["metrics"]
        results['number_of_samples'] = len(data_with_model_predictions)
        results['dataset_size'] = (
            processor._dataset_size
            if processor._dataset_size is not None
            else pred_metadata.get("dataset_size", len(data_with_model_predictions))
        )
        results['task'] = processor.task_type
        results['sub_task'] = processor.sub_task
        results['language'] = processor.language
        logger.info(' ='*30)
        logger.info(f'Model name: {model_name.upper()}')
        logger.info(f'Dataset name: {dataset_name.upper()}')
        for metric in dataset_config["metrics"]:
            metric_score = processor.compute_score(data_with_model_predictions, metrics=metric)
            results.update(metric_score)
            score_val = results[metric]
            logger.info(f"{metric}: {score_val['score'] if isinstance(score_val, dict) else score_val}")
        logger.info(' ='*30)
        logger.info("\n"*3)
        if 'details' in results:
            results['details'] = results['details'][:20]
        with open(score_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    del processor
    from src.model_src.base_model import should_free_model
    if should_free_model():
        logger.info("Memory threshold exceeded — freeing model.")
        if hasattr(model, 'destroy'):
            model.destroy()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return None
    else:
        logger.info("Memory below threshold — keeping model.")
    return model

def main(
        dataset_name      : str  = None,
        model_name        : str  = None,
        batch_size        : int  = 1,     # it is now a dummy parameter
        overwrite         : bool = False,
        metrics           : str  = None,
        number_of_samples : int  = -1,
        log_folder: str = "log_for_all_models",
        backend: str = "transformers",
        min_audio_duration: float = None,
        max_audio_duration: float = None,
        compute_metrics: bool = True,
        skip_inference: bool = False,
    ):

    logger.info(" ="*30)
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Overwrite: {overwrite}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Number of samples: {number_of_samples}")
    logger.info(f"Backend: {backend}")
    logger.info(" ="*30)

    dataset_config = dict(
        number_of_samples=number_of_samples,
        metrics=metrics,
        min_audio_duration=min_audio_duration,
        max_audio_duration=max_audio_duration,
    )
    model_config = dict(batch_size=batch_size, backend=backend)

    run_evaluation(dataset_name, dataset_config, model_name, model_config, None, overwrite, log_folder, compute_metrics, skip_inference)



if __name__ == "__main__":
    fire.Fire(main)
