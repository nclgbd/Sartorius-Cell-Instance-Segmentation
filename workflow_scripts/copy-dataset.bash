#!/bin/bash

printenv | sort

echo "Copying 'sartorius-cell-instance-segmentation.zip' to current directory..."
conda activate skc_env
cp ~/datasets/sartorius-cell-instance-segmentation.zip .
unzip -o -q sartorius-cell-instance-segmentation.zip -d data
rm -rf sartorius-cell-instance-segmentation.zip
ls -la data/
echo "Moving data complete. Exit code $?"