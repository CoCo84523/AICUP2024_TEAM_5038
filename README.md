# TEAM 5038 AI驅動出行未來：跨相機多目標車輛追蹤競賽（模型組）

本程式碼為TEAM 5038用於AICUP競賽<br>


## 安裝

**此程式碼在 Windows 11 上測試**


### 使用 Conda 安裝
**Step 1.** 步驟 1. 創建 Conda 環境並安裝 pytorch
```shell
conda create -n botsort python=3.7
conda activate botsort
```
**Step 2.** 步驟 2. 從 [pytorch.org](https://pytorch.org/get-started/locally/).<br>安裝相應版本的 torch 和 torchvision。<br> 
此程式碼使用 torch 1.11.0+cu113 和 torchvision==0.12.0 進行測試 

**Step 4.** **安裝 numpy**
```shell
pip install numpy
```

**Step 5.** 安裝 requirements.txt 中的依賴項
```shell
pip install -r requirements.txt
```

**Step 6.** 安裝 [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

**Step 7.** 其他依賴項
```shell
# Cython-bbox
pip install cython_bbox

# faiss cpu / gpu
pip install faiss-cpu
pip install faiss-gpu
```

## 數據準備
下載AICUP官方資料集解壓縮並放置到TEAM_5038_AICUP路徑下[訓練集](https://drive.google.com/file/d/1mTDO0SYJ_yzT7PEYYfcCVxsZ1g-AYUgd/view?usp=drive_link) [測試集]()<br>
開啟[資料預處理與模型評估.ipynb](資料預處理與模型評估.ipynb)並由上往下依序執行<br>
為了方便執行後續步驟請手動複製一組images跟labels 並命名為images-example與labels-example
數據的路徑會如下面所示
```
TEAM_5038_AICUP/train/images/0902_150000_151900
TEAM_5038_AICUP/train/images/0902_190000_191900
TEAM_5038_AICUP/test/images/0902_150000_151900
TEAM_5038_AICUP/test/images/0902_190000_191900
```


我們使用的預訓練權重為->[`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)<br>
YOLOv7的預訓練權重請放在以下路徑
```
pretrain_models\yolov7
```
 [YOLOv7_TEAM_5038](https://drive.google.com/file/d/1FBNKWIFTu8516w7yLMpjjhQXtFw6-t8V/view?usp=drive_link)<br>
我們(TEAM_5038)訓練好的YOLOv7權重請放在以下路徑
```
runs\train\TEAM_5038\weights
```
[ReID_TEAM_5038](https://drive.google.com/file/d/1eZsirC9Mj_dD6ZfP8KqoBExC-QSMvSlx/view?usp=sharing)<br>
我們(TEAM_5038)訓練好的ReID權重請放在以下路徑
```
logs\AICUP_115\0509ResNet50_dataProcessed2
```
[MIRNet_TEAM_5038](https://drive.google.com/file/d/1hTiypc1ZsVG2hHnesWIAtXLajx9ME7Lb/view?usp=drive_link)<br>
如有需要 MIRNet的權重請放在以下路徑
```
pretrain_models\MIRNET
```


### 準備 ReID 與 YOLOv7 的數據集 & 轉換AICUP資料到MOT15格式
都包含[資料預處理與模型評估.ipynb](資料預處理與模型評估.ipynb)裡面了<br>
打開來直接依序執行就好
### 訓練 AICUP 的 ReID 模組

```shell
python fast_reid/tools/train_net.py 
```

### 微調 AICUP 的 YOLOv7

``` shell
python yolov7/train.py
```

## 執行追蹤和創建 AICUP 的提交文件（Demo）
對訓練資料做追蹤
輸出路徑會在runs/example
```shell
trake_all_timestamps_train.bat
```
對測試資料做追蹤 輸出路徑會在runs/test
```shell
trake_all_timestamps_test.bat
```
跑完之後回[資料預處理與模型評估.ipynb](資料預處理與模型評估.ipynb)<br>
就可以執行最後一行的模型評估
