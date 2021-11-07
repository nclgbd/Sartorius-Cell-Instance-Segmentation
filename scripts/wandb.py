import wandb
from dotenv import dotenv_values

config = dotenv_values("config/.env")
wandb.init(project=config["project"], 
           entity=config["entity"])
