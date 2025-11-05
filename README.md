# Learning Disentangled Representations via Robust Optimal Transport

**Pytorch** Implementation of **Learning Disentangled Representation via Robust Optimal Transport** 

The implementation is done in pytorch on the **card dataset**.

## Learn shared representation 

To run the first stage of the training, one may use **sdim_trainer.sh**

```
echo Start shared representation training
data_base_folder="data"
xp_name="Share_representation_training"
conf_path="conf/share_conf.yaml"
```
* data_base_folder : This folder should contain the Card dataset in paired-image format. 
  The dataset must be **preprocessed into .h5 files** before training and is **not automatically downloaded**.
  By default, this folder is named "data".

* xp_name : Mlflow experimentation name.

* conf_path : Path to the training configuration file. To use sdim_trainer.sh the conf file must be shaped like **share_conf.yaml** .


## Learn exclusive representation 

To run the first stage of the training, one may use **sdim_trainer.sh** to get pretrained shared encoder and then use
**edim_trainer.sh**.

```
echo Start exclusive representation training
data_base_folder="data"
xp_name="Exclusive_representation_training"
conf_path="conf/exclusive_conf.yaml"
trained_enc_x_path="mlruns/3/38e65dbd8d1246fab33f079e16510019/artifacts/sh_encoder_x/state_dict.pth"
trained_enc_y_path="mlruns/3/38e65dbd8d1246fab33f079e16510019/artifacts/sh_encoder_y/state_dict.pth"
```
* data_base_folder : This folder should contain the Card dataset in paired-image format. 
  The dataset must be **preprocessed into .h5 files** before training and is **not automatically downloaded**.
  By default, this folder is named "data".
  
* xp_name : Mlflow experimentation name.

* conf_path : Path to the training configuration file. To use edim_trainer.sh the conf file must be shaped like **exclusive_conf.yaml**.

* trained_enc_x_path : Path the the pretrained encoder of domains X. As you can see encoders are logged in mlflow.

* trained_enc_y_path : Path the the pretrained encoder of domains Y. As you can see encoders are logged in mlflow.

