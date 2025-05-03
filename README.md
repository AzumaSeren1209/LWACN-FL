# Code for LWACN
## Attribution
This project is based on [PFLlib](https://github.com/TsingZ0/PFLlib), a personalized federated learning library licensed under the Apache License 2.0.
### 🔧 Modifications made:
We have added new methods in the `system/main.py`, `system/flcore/clients/client_avg_mask.py`, `system/flcore/clients/client_avg_mask_adaptive.py`, `system/flcore/servers/server_avg_mask.py`, `system/flcore/servers/server_avg_mask_adaptive.py`, `system/flcore/compresser/mask_decoder.py`, `system/flcore/compresser/mask_encoder.py` directory to support our extensions for [LWACN].
We thank the authors for their excellent work.
### 📜 Licensing
We include the original LICENSE from PFLlib. Please refer to `PFLlib_LICENSE.txt`.
## 🚀 How to Run Our Code
### Environments
Install [CUDA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). 

Install [conda latest](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda. 

For additional configurations, refer to the `prepare.sh` script.  

```bash
conda env create -f env_cuda_latest.yaml  # Downgrade torch via pip if needed to match the CUDA version
```
### Dataset preparation
The code below shows how to generate the MNIST dataset using the util `dataset_utils.py` given in Pfllib.For more available datasets to test, please refer to the `README` in PFLLib:
```bash
cd ./dataset
# Please modify train_ratio and alpha in dataset\utils\dataset_utils.py

python generate_MNIST.py iid - - # for iid and unbalanced scenario
python generate_MNIST.py iid balance - # for iid and balanced scenario
python generate_MNIST.py noniid - pat # for pathological noniid and unbalanced scenario
python generate_MNIST.py noniid - dir # for practical noniid and unbalanced scenario`
python generate_MNIST.py noniid - exdir # for Extended Dirichlet strategy 
```
### Our code
1. To run our extended code based on PFLlib, please use the following command format:
```bash
conda activate LWACN
python main.py --device_id 2 --model MobileNet --algorithm FedAvg --dataset CIFAR100_noniiddir --global_rounds 3000 --save_name CIFAR100_Mobile_dir_FedAvg
python main.py --device_id 2 --model MobileNet --algorithm FedAvgMask --dataset CIFAR100_noniiddir --global_rounds 3000 --save_name CIFAR100_Mobile_dir_0.8 --min_compress_rate 0.8
python main.py --device_id 2 --model MobileNet --algorithm FedAvgMask --dataset CIFAR100_noniiddir --global_rounds 3000 --save_name CIFAR100_Mobile_dir_0.9 --min_compress_rate 0.9
python main.py --device_id 2 --model MobileNet --algorithm FedAvgMask --dataset CIFAR100_noniiddir --global_rounds 3000 --save_name CIFAR100_Mobile_dir_0.95 --min_compress_rate 0.95
python main.py --device_id 2 --model MobileNet --algorithm FedAvgMaskAdaptive --dataset CIFAR100_noniiddir --global_rounds 3000 --save_name CIFAR100_Mobile_dir_Adaptive
```
2. To adjust the hyperparameters (such as $\alpha,\beta$, etc.), please go to `PFLlib\system\flcore\clients\client_avg_mask_adaptive.py` and change the hyperparameters given in the list:`self.layer_importance_w`,`self.mu_w`.
