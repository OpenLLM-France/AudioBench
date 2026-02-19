
import os
import fire
import json
import logging
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

    # if batch_size > 1 :
    #     raise NotImplementedError("Batch size {} not implemented yet".format(batch_size))
    
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
        dataset_name      : str  = None,
        model_name        : str  = None,
        batch_size        : int  = 1,     # it is now a dummy parameter
        overwrite         : bool = False,
        metrics           : str  = None,
        number_of_samples : int  = -1,
        log_folder: str = "log_for_all_models",
        backend: str = "transformers"
    ):
    
    processor = load_dataset_processor(dataset_name, number_of_samples)

    if metrics is None:
        if processor.metrics is not None:
            metrics = processor.metrics
            logger.info(f"Metrics is not specified. Use the default metrics of the dataset: {metrics}")
        else:
            raise NotImplementedError(f"The dataset {dataset_name} does not have a default metrics.")

    # If the final score log exists, skip the evaluation (no data loading needed)
    if not overwrite and os.path.exists(f'{log_folder}/{model_name}/{dataset_name}_{metrics}_score.json'):
        logger.info("Evaluation has been done before. Skip the evaluation.")
        with open(f'{log_folder}/{model_name}/{dataset_name}_{metrics}_score.json', 'r') as f:
            results = json.load(f)
        logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
        logger.info(f'Dataset name: {dataset_name.upper()}')
        logger.info(f'Model name: {model_name.upper()}')
        logger.info(json.dumps({metrics: results[metrics]}, indent=4, ensure_ascii=False))
        logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

        logger.info("\n\n\n\n\n")
        return

    if model_name == 'WavLLM_fairseq':
        batch_size = -1
        logger.info("Batch size is set to -1 for WavLLM_fairseq model.")

    if overwrite or not os.path.exists(f'{log_folder}/{model_name}/{dataset_name}.json'):
        logger.info(f'Overwrite is enabled or the results are not found. Try to infer with the model: {model_name}.')

        # Load dataset (deferred until now to skip download when not needed)
        processor.load()
        input_data = processor.prepare_model_input()

        # Load model
        model = load_model(model_name, backend=backend)

        # Specific current dataset name for evaluation
        model.dataset_name = dataset_name

        # Infer with model
        model_predictions = do_model_prediction(input_data, model, batch_size=batch_size)
        data_with_model_predictions = processor.format_model_predictions(input_data, model_predictions)

        # Save the result with predictions
        os.makedirs(f'{log_folder}/{model_name}', exist_ok=True)
        with open(f'{log_folder}/{model_name}/{dataset_name}.json', 'w') as f:
            json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)

    data_with_model_predictions = json.load(open(f'{log_folder}/{model_name}/{dataset_name}.json'))

    results = processor.compute_score(data_with_model_predictions, metrics=metrics)

    # Take only the first 100 samples for record.
    if 'details' in results:
        results['details'] = results['details'][:20]

    # Print the result with metrics
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    logger.info(f'Dataset name: {dataset_name.upper()}')
    logger.info(f'Model name: {model_name.upper()}')
    logger.info(json.dumps({metrics: results[metrics]}, indent=4, ensure_ascii=False))
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

    # Save the scores
    with open(f'{log_folder}/{model_name}/{dataset_name}_{metrics}_score.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)   

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

    logger.info("= = "*20)
    logger.info("Dataset name: {}".format(dataset_name))
    logger.info("Model name: {}".format(model_name))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Overwrite: {}".format(overwrite))
    logger.info("Metrics: {}".format(metrics))
    logger.info("Number of samples: {}".format(number_of_samples))
    logger.info("Backend: {}".format(backend))
    logger.info("= = "*20)

    run_evaluation(dataset_name, model_name, batch_size, overwrite, metrics, number_of_samples, log_folder, backend)
    


if __name__ == "__main__":
    fire.Fire(main)
