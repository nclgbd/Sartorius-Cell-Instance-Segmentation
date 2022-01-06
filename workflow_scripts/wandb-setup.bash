echo "wandb_resume=never
wandb_mode=online
wandb_job_type=$1
wandb_tags=baseline
wandb_entity=nclgbd
wandb_project=Sartorius-Kaggle-Competition
wandb_api_key=$2" >>"$3"
cat "$3"
