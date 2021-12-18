#!/bin/sh
touch config/.env


echo 'project="Sartorius-Cell-Instance-Segmentation"' >> config/.env
echo 'entity="${{secrets.ENTITY}}"' >> config/.env
echo 'wandb_api_key=${{secrets.WANDB_API_KEY}}' >> config/.env

echo "config/.env contents:"
cat config/.env