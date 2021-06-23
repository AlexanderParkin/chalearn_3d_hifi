# Chalearn 3D High-Fidelity Mask Face Presentation Attack Detection Challenge
This is code of our solution for Chalearn 3D High-Fidelity Mask Face Presentation Attack Detection Challenge at ICCV 2021.
Our solution based on analyzing parts of the face in search of small parts that would give out attack attempts of high-quality 3D masks.

![net](https://user-images.githubusercontent.com/11870868/122923804-bbc3cf00-d36d-11eb-9909-4d70286af836.jpg)




## Training steps
### Step 1.
Install at_learner_core
```bash
conda create -n competition_env python=3.9
conda activate competition_env
cd path/to/chalearn_3d_hifi
pip install -e at_learner_core/
```

### Step 2.
Launch the open-source face detector on the entire training set to get a bbox of the largest face for train/val/test data. 
You can skip this step, since we have done it with train_out.csv, val_out.csv, test_out.csv and also posted the results.

```bash
# Run face detectors and save results
python make_dataset_detections.py --data_dir /path/to/img/folder/ --test_list test.txt
    --train_list train_label.txt --val_list val_label.txt --test_list_out test_out.csv --train_list_out train_out.csv
    --val_list_out val_out.csv
```

### Step 3.
Make crops from initial images and detections.

```bash
# Run face detectors and save results
python make_dataset_crops.py --data_dir /path/to/img/folder/
    --test_list_out test_out.csv --train_list_out train_out.csv
    --val_list_out val_out.csv --crops_dir /path/to/crops/folder/
```

### Step 4.
Run train process
```bash
cd ../casia_track
python save_config_crops.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config experiments/CASIA_Hifi/exp21/CASIA_Hifi_exp21.config;
```

### Step 5.
Run inference
```bash
bash inference.sh
```

You can skip Step 3 and run inference on our checkpoint. Trained model download link:
```
pip install gdown
gdown https://drive.google.com/uc?id=1wdSGG_5JBCj0Ffldjq4P0_CA7Xz0sq5c
```
