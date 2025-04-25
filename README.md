# Introduction
![figure1](https://github.com/YidongSong/InterAb/blob/main/Model%20architecture.png)

# System requirement
InterAb is developed under Linux environment with:
Python 3.8.16, numpy v1.24.3, pyg v2.3.0, pytorch v1.13.1, biopython v1.83, debugpy v1.6.7, decorator v5.1.1, filelock, v3.12.1, gmp v6.2.1, idna v3.4, ipython v8.12.0, openfold v1.0.1, scipy 1.10.1, and six v1.16.0

# Install and run the program
**1.** Clone this repository by 'git clone https://github.com/YidongSong/InterAb.git'.

**2.** Install the packages required by InterAb.
```
conda create -n <env_name> python==3.8
conda activate <env_name>
conda install <the aforementioned packages>
```

**3.** Download the models.
Download the model from [Zenodo](https://zenodo.org/uploads/15278695), which includes the pre-trained antibody models (Antibody_models), the pre-trained ESM2 models (ESM_models), and the trained InterAb model. You can also download the ESM2 model from [huggingface](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main). The downloaded models are stored in the following locations:
```
Antibody_models -> InterAb/Antibody_models
ESM_models -> InterAb/ESM_models
model -> InterAb/model
```

**4.** Run InterAb with the following command:  
For the task of antibody specificity prediction, please use the following code:
```
bash run.sh ./config/common/specificity.json specificity
```
The `specificity.json` documents the parameters of the model, and `specificity` indicates the task to be predicted.   

For the task of antibody-antigen affinity prediction, please use the following code:

```
bash run.sh ./config/common/affinity.json affinity
```
Among them, `affinity.json` represents the parameters of the model, and `affinity` denotes the type of task.

**5.** Analysis of the prediction results.
The prediction results are stored in `./Results`, encompassing the predictions for both antibody specificity and antibody-antigen affinity.  
For antibody specificity prediction, 预测结果格式为
## 预测结果

以下是模型预测结果的详细说明：

| Heavy chain          | Light chain                         | Antigen                     |Predictions
|-------------------|-------------------------------|------------------------------|
| QVQLQQ******KISCKS | ISCKTS******VDKPGQ                  | 预测任务（antibody specificity） |
| `affinity.json`    | 记录抗体-抗原亲和力预测结果   | 抗体-抗原亲和力（antibody-antigen affinity） |
| `results.csv`      | 存储所有预测结果的CSV文件     | 包含所有预测结果的汇总       |
