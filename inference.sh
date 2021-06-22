#!/bin/bash
GPU_ID=0

python test_config.py
CUDA_VISIBLE_DEVICES=$GPU_ID python casia_predictor.py --test_config experiment_tests/test/test.config \
 --model_config_path experiments/CASIA_Hifi/exp21/CASIA_Hifi_exp21.config \
 --checkpoint_path experiments/CASIA_Hifi/exp21/checkpoints/model_59.pth
 CUDA_VISIBLE_DEVICES=$GPU_ID python casia_predictor.py --test_config experiment_tests/test_flip/test_flip.config \
 --model_config_path experiments/CASIA_Hifi/exp21/CASIA_Hifi_exp21.config \
 --checkpoint_path experiments/CASIA_Hifi/exp21/checkpoints/model_59.pth

python compile_submit_file.py --output_scores experiment_tests/test/CASIA_Hifi/exp21/TestFileLogger/output_59.txt \
--hflip_output_scores experiment_tests/test_flip/CASIA_Hifi/exp21/TestFileLogger/output_59.txt