#!/bin/bash
xai_method=gradcam
python avg_heatmaps.py \
	--heatmap-path lightning_logs/version_0/checkpoints/$xai_method.h5 \
	lightning_logs/version_1/checkpoints/$xai_method.h5 \
	lightning_logs/version_2/checkpoints/$xai_method.h5 \
	lightning_logs/version_3/checkpoints/$xai_method.h5 \
	lightning_logs/version_4/checkpoints/$xai_method.h5 \
	lightning_logs/version_5/checkpoints/$xai_method.h5 \
	lightning_logs/version_6/checkpoints/$xai_method.h5 \
	lightning_logs/version_7/checkpoints/$xai_method.h5 \
	lightning_logs/version_8/checkpoints/$xai_method.h5 \
	lightning_logs/version_9/checkpoints/$xai_method.h5 \
	--out-dir average-heatmaps/ens0/
