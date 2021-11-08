import os
import wandb
from dotenv import dotenv_values

config = dotenv_values("config/.env")
os.environ['WANDB_API_KEY'] = config["wandb_api_key"]

with wandb.init(project=config["project"], entity=config["entity"], reinit=True) as run:
    for x in range(10):
        for y in range(100):
            run.log({"metric": x+y})
