# Chalearn 3D High-Fidelity Mask Face Presentation Attack Detection Challenge
This is code of our solution for Chalearn 3D High-Fidelity Mask Face Presentation Attack Detection Challenge at ICCV 2021.
Our solution based on analyzing parts of the face in search of small parts that would give out attack attempts of high-quality 3D masks.

## Training steps
### Step 1.
Install at_learner_core
```bash
cd /path/to/new/pip/environment
python -m venv competition_env
source competition_env/bin/activate
pip install -r requirements.txt
pip install -e at_learner_core/
```

### Step 2.
Launch the open-source face detector on the entire training set to get a bbox of the largest face for train/val/test data. 
You can skip this step, since we have done it with train.csv, val.csv, test.csv and also posted the results.

```bash
# Run face detectors and save results
python make_dataset.py
```

### Step 3.
Run train process
```bash
cd ../casia_track
python save_config_crops.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config experiments/CASIA_Hifi/exp21/CASIA_Hifi_exp21.config;
```

### Step 4.
Run inference
```bash
python test_config.py
CUDA_VISIBLE_DEVICES=0 python casia_predictor.py --test_config experiment_tests/test/test.config \
 --model_config_path experiments/CASIA_Hifi/exp21/CASIA_Hifi_exp21.config \
 --checkpoint_path experiments/CASIA_Hifi/exp21/checkpoints/model_59.pth
```

### Step 5.
Compile submit_file
```bash
python compile_submit_file.py
```