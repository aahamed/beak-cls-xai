#!/bin/bash
xai_method=guided-gradcam
ens_name=ens4
for ((i=0; i<1;i+=1))
do
	python gen_heatmaps.py \
		--model-path=lightning_logs/$ens_name/version_$i/checkpoints/best_model.ckpt \
		--xai-method $xai_method \
		--batch-size=1
done
# python gen_heatmaps.py --model-path=lightning_logs/version_0/checkpoints/best_model.ckpt
#python gen_heatmaps.py \
#	--model-path=lightning_logs/version_$i/checkpoints/best_model.ckpt \
#	--xai-method integrated-gradients --batch-size=4
