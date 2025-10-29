from pathlib import Path
import mteb
import argparse

import dotenv
dotenv.load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import collector module and set module-level variables
import mteb.collector as collector_module
from mteb.collector import get_collector

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MTEB for a given task and save results to JSONL in `ze_results` format")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the MTEB dataset to evaluate"
    )
    parser.add_argument(
        "--dump-file", 
        "-d",
        type=str,
        help="Path to JSONL file to dump collected data"
    )
    parser.add_argument(
        "--overwrite", 
        "-o",
        action="store_true",
        help="Overwrite existing dump file (rather than append)"
    )
    args = parser.parse_args()

    print(f"Value of overwrite: {args.overwrite}")

    # load environment variables
    dotenv.load_dotenv()

    # Initialize MTEB task
    model_name = "openai/text-embedding-3-small"
    model = mteb.get_model(model_name)

    # Set module-level variables in collector module
    collector_module.DATASET_NAME = args.dataset_name
    task = mteb.get_task(collector_module.DATASET_NAME)

    collector_module.SCORING_METRIC = task.metadata.main_score

    #########################################
    # RUN THE MTEB TASK
    #########################################

    collector = get_collector()

    results = mteb.evaluate(model, tasks=task, overwrite_strategy="always")
    
    print(results.to_dataframe())

    collector.prepare()
    if args.dump_file:
        dump_path = Path(args.dump_file)
        collector.dump_to_json(dump_path, overwrite=args.overwrite)