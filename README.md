# Chalearn 3D High-Fidelity Mask Face Presentation Attack Detection Challenge
This is code of our solution for Chalearn 3D High-Fidelity Mask Face Presentation Attack Detection Challenge at ICCV 2021.
Our solution based on analyzing parts of the face in search of small parts that would give out attack attempts of high-quality 3D masks.

![net](https://user-images.githubusercontent.com/11870868/122923804-bbc3cf00-d36d-11eb-9909-4d70286af836.jpg)




## Training steps
### Step 1.
Install at_learner_core
```bash
cd /path/to/new/pip/environment
python -m venv competition_env
source competition_env/bin/activate
pip install -e at_learner_core/
```

### Step 2.
Launch the open-source face detector on the entire training set to get a bbox of the largest face for train/val/test data. 
You can skip this step, since we have done it with train.csv, val.csv, test.csv and also posted the results.

```bash
# Run face detectors and save results
python make_dataset.py --data_dir /path/to/img/folder/ --list_dir /path/to/lists/dir --test_list test.txt
    --train_list train.txt --val_list val.txt --test_list_out test_out.csv --train_list_out train_out.csv
    --val_list_out val_out.csv --crops_dir /path/to/crops/folder/
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
bash inference.sh
```

Trained model download link:
```
pip install gdown
gdown https://drive.google.com/uc?id=1wdSGG_5JBCj0Ffldjq4P0_CA7Xz0sq5c
```
