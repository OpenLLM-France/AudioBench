
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

    if batch_size > 1 and not model.supports_vllm:
        raise NotImplementedError(f"Batch size {batch_size} not implemented yet for {model} (vllm support: {model.supports_vllm})")

    if batch_size > 1:
        model_predictions = model.generate(input_data)

    else:
        model_predictions = []
        for inputs in tqdm(input_data, leave=False):
            outputs = model.generate(inputs)
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

    if dataset_config.get("metrics") is None:
        if processor.metrics is not None:
            dataset_config["metrics"] = processor.metrics
            logger.info(f"Metrics is not specified. Use the default metrics of the dataset: {dataset_config.get('metrics')}")
        else:
            raise NotImplementedError(f"The dataset {dataset_name} does not have a default metrics.")

    model_dir = Path(log_folder) / model_name
    score_path = model_dir / f"{dataset_name}_{dataset_config["metrics"]}_score.json"
    prediction_path = model_dir / f"{dataset_name}.json"


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
        logger.info('- '*30)
        logger.info(f'Model name: {model_name.upper()}')
        logger.info(f'Dataset name: {dataset_name.upper()}')
        logger.info(f"Evaluation for {model_name} and {dataset_name} exists. Skip the evaluation.")
        logger.info(json.dumps({dataset_config["metrics"]: results[dataset_config["metrics"]]}, indent=4, ensure_ascii=False))
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
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(prediction_path, 'w') as f:
            json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)

    data_with_model_predictions = json.loads(prediction_path.read_text())

    results = processor.compute_score(data_with_model_predictions, metrics=dataset_config["metrics"])

    if 'details' in results:
        results['details'] = results['details'][:20]
        
    results['model_name'] = model_name
    results['dataset_name'] = dataset_name
    results['metrics'] = dataset_config["metrics"]
    results['number_of_samples'] = len(data_with_model_predictions)
    results['task'] = processor.task_type

    # Print the result with metrics
    logger.info(' ='*30)
    logger.info(f'Model name: {model_name.upper()}')
    logger.info(f'Dataset name: {dataset_name.upper()}')
    logger.info(json.dumps({dataset_config["metrics"]: results[dataset_config["metrics"]]}, indent=4, ensure_ascii=False))
    logger.info(' ='*30)
    logger.info("\n"*3)

    # Save the scores
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
