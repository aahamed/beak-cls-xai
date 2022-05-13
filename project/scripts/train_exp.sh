#!/bin/bash
python beak_classifier.py --max_epochs=20 --accelerator=gpu --devices=1 --seed=1111 --backbone=resnet18 --weight_decay=1e-3 --batch_size=64 --tune_mode=fine-tune --data_method=full
#python beak_classifier.py --max_epochs=20 --accelerator=gpu --devices=1 --seed=1111 --backbone=resnet18 --weight_decay=0 --batch_size=64 --tune_mode=fine-tune --data_method=bagging
#python beak_classifier.py --max_epochs=20 --accelerator=gpu --devices=1 --seed=1111 --backbone=resnet18 --weight_decay=0 --batch_size=64 --tune_mode=fine-tune --data_method=kfold
#python beak_classifier.py --max_epochs=20 --accelerator=gpu --devices=1 --seed=1111 --backbone=resnet18 --weight_decay=1e-4 --batch_size=64 --tune-mode=fine-tune
#python beak_classifier.py --max_epochs=20 --accelerator=gpu --devices=1 --seed=1111 --backbone=resnet18 --weight_decay=1e-3 --batch_size=64 --tune-mode=fine-tune
#python beak_classifier.py --max_epochs=20 --accelerator=gpu --devices=1 --seed=1111 --backbone=resnet50 --weight_decay=1e-3 --batch_size=64 --tune-mode=fine-tune
