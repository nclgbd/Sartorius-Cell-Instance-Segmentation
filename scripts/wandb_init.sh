#!/bin/bash

touch scripts/.env

{
echo 'project="Sartorius-Cell-Instance-Segmentation"'
'entity="{{secrets.ENTITY}}"'
'wandb_api_key="{{secrets.WANDB_API_KEY}}"' >> scripts/.env
} >> scripts/.env

echo "scripts/.env contents:"
cat scripts/.env
python scripts/wandb_init.py

