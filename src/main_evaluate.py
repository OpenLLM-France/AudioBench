
import fire
import json
import logging
from pathlib import Path
from tqdm import tqdm

from src.dataset_factory import load_dataset_processor
from model_factory import load_model

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


def do_model_prediction(input_data, model, batch_size):
    if model.backend=="vllm":
        return model.generate(input_data)

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
    ):
    
    if not dataset_config.get("path"):
        processor = load_dataset_processor(dataset_name, dataset_config.get("number_of_samples"))
    else:
        processor = load_dataset_processor(dataset_config.get("path"), dataset_config.get("number_of_samples"))
    
    if dataset_config.get("task") is not None:
        processor.task_type = dataset_config.get("task").upper()
        
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

    if model_name == 'WavLLM_fairseq':
        model_config["batch_size"] = -1
        logger.info("Batch size is set to -1 for WavLLM_fairseq model.")

    if not overwrite and prediction_path.exists():
        predictions = json.loads(prediction_path.read_text())
        if dataset_config.get("number_of_samples")>0 and len(predictions)<dataset_config.get("number_of_samples"):
            overwrite = True
            logger.info(f"Found {len(predictions)} samples in {prediction_path} instead of {dataset_config.get("number_of_samples")}. Overwrite set to True.")
    
    if not overwrite and score_path.exists():
        results = json.loads(score_path.read_text())
        if (all([metric in results for metric in dataset_config["metrics"]])):
            logger.info('- '*30)
            logger.info(f"Evaluation for {model_name.upper()} and {dataset_name.upper()} exists. Skip the evaluation.")
            logger.info(results)
            logger.info('- '*30)
            logger.info("\n"*3)
            return model

    if overwrite or not prediction_path.exists():
        logger.info(f'Overwrite is enabled or the results are not found. Try to infer with the model: {model_name}.')

        # Load dataset (deferred until now to skip download when not needed)
        processor.load()
        input_data = processor.prepare_model_input()

        # Load model
        if model is None:
            model = load_model(model_name, backend=model_config["backend"])

        # Specific current dataset name for evaluation
        model.dataset_name = dataset_name

        # Infer with model
        model_predictions = do_model_prediction(input_data, model, batch_size=model_config["batch_size"])
        data_with_model_predictions = processor.format_model_predictions(input_data, model_predictions)

        # Save the result with predictions
        with open(prediction_path, 'w') as f:
            json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)

    data_with_model_predictions = json.loads(prediction_path.read_text())
    results = dict()
    results['model_name'] = model_name
    results['dataset_name'] = dataset_name
    results['metrics'] = dataset_config["metrics"]
    results['number_of_samples'] = len(data_with_model_predictions)
    results['task'] = processor.task_type
    results['language'] = processor.language
    logger.info(' ='*30)
    logger.info(f'Model name: {model_name.upper()}')
    logger.info(f'Dataset name: {dataset_name.upper()}')
    for metric in dataset_config["metrics"]:
        metric_score = processor.compute_score(data_with_model_predictions, metrics=metric)
        results.update(metric_score)
        logger.info(f"{metric}: {results[metric]}")
    logger.info(' ='*30)
    logger.info("\n"*3)
    if 'details' in results:
        results['details'] = results['details'][:20]
    with open(score_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return model

def main(
        dataset_name      : str  = None,
        model_name        : str  = None,
        batch_size        : int  = 1,     # it is now a dummy parameter
        overwrite         : bool = False,
        metrics           : str  = None,
        number_of_samples : int  = -1,
        log_folder: str = "log_for_all_models",
        backend: str = "transformers"
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
    
    dataset_config = dict(number_of_samples=number_of_samples, metrics=metrics)
    model_config = dict(batch_size=batch_size, backend=backend)

    run_evaluation(dataset_name, dataset_config, model_name, model_config, overwrite, log_folder)



if __name__ == "__main__":
    fire.Fire(main)
