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

