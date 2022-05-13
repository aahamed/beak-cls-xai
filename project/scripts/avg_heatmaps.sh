#!/bin/bash
xai_method=guided-gradcam
ens_name=ens4
python avg_heatmaps.py \
	--heatmap-path lightning_logs/$ens_name/version_0/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_1/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_2/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_3/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_4/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_5/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_6/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_7/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_8/checkpoints/$xai_method.h5 \
	lightning_logs/$ens_name/version_9/checkpoints/$xai_method.h5 \
	--out-dir average-heatmaps/$ens_name/
