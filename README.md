

## 💻 Installation

If you want to building from source, here is a quick installation example : 
```
conda create --name mobileseed python=3.8 -y
conda activate mobileseed

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install mmengine
mim install "mmcv-full>=1.6.0,<1.8.0"

pip install -r requirements.txt

cd Mobile-Seed
pip install -v -e .
```

## 🚅 Usage
### Evaluation
```
This is our test dataset: 

Unzip the provided VOCdevkit.zip(download here: https://whueducn-my.sharepoint.com/:f:/g/personal/martin_liao_whu_edu_cn/EjklDmgVOitPrhuAwy6h6EkBPkyTvnlCkTN0BdjPIIc6xA?e=1i6D4Z ) file and place it in the 'data/' directory. Unzip test_image.zip file and place it in the 'data/test' directory.



Place MS_tiny_pascal_context.pth, iter_1000.pth and latest.pth in the ckpt folder (create it in the root directory if it doesn't exist).



"PASCAL dataset": ./data/test/test_image
"Flower and Bird dataset": ./data/VOCdevkit/VOC2010

Example: evaluate  ```MS_BR ``` on  ```Flower-Birds dataset```:

python inference.py \
  --config configs/Mobile_Seed/MS_tiny_pascal_context.py \
  --checkpoint /root/Mobile-Seed/ckpt/MS_tiny_pascal_context.pth \
  --input path/to/test_images/ \
  --output path/to/save_results/ \
  --opacity 0.6
```
```
--config	str	configs/.../MS_tiny_pascal_context.py	    '''Path to model configuration file'''
--checkpoint	str	.../iter_1000.pth	                '''Path to the model weight file (.pth)'''
--input	str	.../data/test/test_image/	                 '''Input folder path for the images to be tested'''
--output	str	.../results_MS_BR_FBDataset/	'''Path to the folder where the inference results are saved'''
--opacity	float	0.5	                             '''Visualize the mask's transparency (0.0 - 1.0)'''
--num_classes	int	60	                                    '''Number of categories in the dataset'''
```



### 🔦 Training

Example: train ```MS_BR``` on ```PASCAL dataset```:

# Single-gpu training
```
python tools/train.py configs/Mobile_Seed/MS_tiny_pascal_transfer.py
```

# Multi-gpu training
```
python tools/train.py configs/Mobile_Seed/MS_tiny_pascal_transfer.py <GPU_NUM>
```


