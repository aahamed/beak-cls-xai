#!/bin/bash
for ((i=0; i<10;i+=1))
do
	python gen_heatmaps.py \
		--model-path=lightning_logs/version_$i/checkpoints/best_model.ckpt \
		--xai-method gradcam
	python gen_heatmaps.py \
		--model-path=lightning_logs/version_$i/checkpoints/best_model.ckpt \
		--xai-method guided-gradcam
	#python gen_heatmaps.py \
	#	--model-path=lightning_logs/version_$i/checkpoints/best_model.ckpt \
	#	--xai-method integrated-gradients
	python gen_heatmaps.py \
		--model-path=lightning_logs/version_$i/checkpoints/best_model.ckpt \
		--xai-method saliency
	#python gen_heatmaps.py \
	#	--model-path=lightning_logs/version_$i/checkpoints/best_model.ckpt \
	#	--xai-method lime
done
# python gen_heatmaps.py --model-path=lightning_logs/version_0/checkpoints/best_model.ckpt
