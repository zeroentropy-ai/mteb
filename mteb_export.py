import mteb
print(mteb.__file__)

import dotenv
dotenv.load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

from mteb.utils import ROOT, ZeroModel

USE_OPENAI_EMBEDDINGS = True
CURRENT_TASK = "StackOverflowQA"

# Create model, default to ZeroModel when testing
model = ZeroModel()
if USE_OPENAI_EMBEDDINGS:
    model = mteb.get_model("openai/text-embedding-3-small")

# Load task
task = mteb.get_task(CURRENT_TASK)

# Evaluate
results = mteb.evaluate(model, task, encode_kwargs={"batch_size": 128})

# Print results
print(results.to_dataframe())