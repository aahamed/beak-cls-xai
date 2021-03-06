#!/bin/bash
data_method=bagging
for ((i=0; i<10;i+=1))
do
	python beak_classifier.py --max_epochs=20 --accelerator=gpu --devices=1 \
		--seed=$i --backbone=resnet18 --weight_decay=1e-3 --batch_size=64 \
		--tune_mode=fine-tune --data_method=$data_method
done
# python beak_classifier.py --max_epochs=6 --accelerator=gpu --devices=1 --resume_from_checkpoint=./lightning_logs/version_1/checkpoints/epoch\=5-step\=942.ckpt
