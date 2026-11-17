# Introduction
![figure1](https://github.com/YidongSong/InterAb/blob/main/Model%20architecture.png)
InterAb is a novel model developed for predicting antibody-antigen interactions and optimizing antibodies through all-atom modeling and antibody language models. Leveraging the proposed all-atom modeling approach, AtomInter, and pre-trained antibody language models, InterAb outperforms existing methods in predicting antibody specificity and antibody-antigen binding affinity. Furthermore, it has been successfully applied to the optimization of broadly neutralizing antibodies.
# System requirement
InterAb is developed under Linux environment with:
Python 3.8.20, numpy v1.24.3, pyg v2.3.0, pytorch v2.0.0, biopython v1.81, debugpy v1.6.7, decorator v5.1.1, filelock, v3.12.1, gmp v6.3.0, idna v3.10, ipython v8.12.0, scipy 1.10.1, and six v1.16.0.

Detailed information can be found in environment.yml.

# Install and run the program
**1.** Clone this repository by 'git clone https://github.com/YidongSong/InterAb.git'.

**2.** Install the packages required by InterAb.
```
# manually installed individually
conda create -n <env_name> python==3.8
conda activate <env_name>
conda install <the packages in the environment.yml>

# automatically installed
conda env create -f environment.yml
```

**3.** Download the models.
Download the model from [Zenodo](https://zenodo.org/records/21785933), which includes the pre-trained antibody models (Antibody_models), the pre-trained ESM2 models (ESM_models), and the trained InterAb model. You can also download the ESM2 model from [huggingface](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main). The downloaded model, after being decompressed, should be used to replace the files below:
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
The prediction results are stored in `./Results/pred`, encompassing the predictions for both antibody specificity and antibody-antigen affinity.  
For antibody specificity prediction, the format of the prediction results is as follows:


| Heavy chain          | Light chain                         | Antigen                     |Predictions      |
|:-------------------:|:-------------------------------:|:------------------------------:|:------------------------------:|
| QVQLQQ......KISCKS | ISCKTS......VDKPGQ                  | QNKKWL......LRSLVA |0.0002636 |
| ASGFTV......YMSWVR    | LNWYQQ......GKAPKL   | AAAVKQ......EEGICG |0.99756855 |
| APGKGL......VAYIYP      | CRASQS......SVSSAV     | HVASGY......EAEVIP       |0.9999988 |

The `Heavy chain` and `Light chain` represent the heavy chain and light chain of the antibody, while the `Antigen` denotes the chain of the antigen. The `Predictions` denote the prediction scores, where a higher score indicates a greater likelihood of binding between the corresponding antibody and antigen.   


The prediction results for antibody-antigen affinity are presented in the following format:

| Heavy chain          | Light chain                         | Antigen                     |Predictions      |
|:-------------------:|:-------------------------------:|:------------------------------:|:------------------------------:|
|SPRLLI......ASQSIG| WIRKFP......GNKLEY|DNYRGY......SLGNWV|-9.820102|
|PREEQY......STYRVV|PEVKFN......NWYVDG|FHNESL......SSQASS|-9.166173|
|FTFSRY......WVRQAP|CSASSS......VHMFWY|IAFLND......KRMDIG|-9.843646|

Similarly, `Heavy chain`, `Light chain`, and `Antigen` represent the sequences of the antibody and antigen, respectively, while `Predictions` denote the predicted binding free energy (Delta G), where a lower Delta G value indicates a higher affinity.


# Data availability
The specificity dataset SPE7620 and the affinity dataset AFF1728 are stored in `./data`. The dataset curated from the SKEMPI 2.0 database is located in `./data/SKE426.csv`, and the organized data for SARS-CoV-2 and its variants are stored in `./data/SARS-CoV-2`.

# Citation and contact
```
@article{song2026optimizing,
  title={Optimizing broadly neutralizing antibodies via all-atom interaction modeling and pre-trained language models},
  author={Song, Yidong and Wu, Fandi and Wang, Rubo and He, Bing and Yan, Qihong and Huang, Xiaohan and Chen, Sheng and Yuan, Qianmu and Rao, Jiahua and Tang, Zhenchao and others},
  journal={bioRxiv},
  pages={2026--01},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

In case you have questions, please contact Yidong Song (laliofchina@gmail.com).  





