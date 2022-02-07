
# Linking Physics-based Imaging to Data-driven Learning for Underwater Image Clearness

We implement our method with PyTorch on a GEFORCE RTX 3090 GPU.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Datasets
- UIEB: [https://li-chongyi.github.io/proj_benchmark.html](https://li-chongyi.github.io/proj_benchmark.html)
- EUVP: [http://irvlab.cs.umn.edu/resources/euvp-dataset/](http://irvlab.cs.umn.edu/resources/euvp-dataset)
- MIT-Adobe 5K: [https://data.csail.mit.edu/graphics/fivek/](https://data.csail.mit.edu/graphics/fivek/)

## UIC under Ideal Supervision
- Experiments on UIEB dataset
```bash
# Train
CUDA_VISIBLE_DEVICES=0 python train.py  --model Ideal --name Ideal_uieb --dataset_root uieb_dataset_url --dataset_mode uieb --batch_size 8
# Test
CUDA_VISIBLE_DEVICES=0 python test.py --model Ideal --name Ideal_uieb --dataset_root uieb_dataset_url --dataset_mode uieb --batch_size 1
# Metrics: MSE, PSNR, SSIM
CUDA_VISIBLE_DEVICES=0 python metrics/evaluations-enhancement.py --results_root results/Ideal_uieb/test_latest/images/ --dataname uieb
```
- Experiments on EUVP dataset
```bash
# Train
CUDA_VISIBLE_DEVICES=0 python train.py  --model Ideal --name Ideal_euvp --dataset_root euvp_paired_dataset_url --dataset_mode euvp --batch_size 8
# Test
CUDA_VISIBLE_DEVICES=0 python test.py --model Ideal --name Ideal_euvp --dataset_root euvp_paired_dataset_url --dataset_mode euvp --batch_size 1
# Metrics: MSE, PSNR, SSIM
CUDA_VISIBLE_DEVICES=0 python metrics/evaluations-enhancement.py --results_root results/Ideal_euvp/test_latest/images/ --dataname euvp
```
## UIC under Non-ideal In-water Supervision

```bash
# Train
CUDA_VISIBLE_DEVICES=0 python train.py --model NonIdealGAN --name NonIdealGAN_euvp --dataset_root euvp_unpaired_dataset_url --unaligned_dataset euvp --batch_size 8
# Test
CUDA_VISIBLE_DEVICES=0 python test.py --model NonIdealGAN --name NonIdealGAN_euvp --dataset_root euvp_paired_dataset_url --dataset_mode euvp --batch_size 1
# Metrics: MSE, PSNR, SSIM
CUDA_VISIBLE_DEVICES=0 python metrics/evaluations-enhancement.py --results_root results/NonIdealGAN_euvp/test_latest/images/ --dataname euvp
# Metrics: UIQM
python metrics/uiqm_metrics.py  --results_root  results/NonIdealGAN_euvp/test_latest/images/
```

## UIC under Non-ideal In-air Supervision

```bash
# Train
CUDA_VISIBLE_DEVICES=0 python train.py --model NonIdealRAS --name NonIdealRAS_In-air --dataset_root euvp_unpaired_dataset_url  --unaligned_dataset adobe5k --batch_size 8
# Test
CUDA_VISIBLE_DEVICES=0 python test.py --model NonIdealRAS --name NonIdealRAS_In-air --dataset_root euvp_paired_dataset_url --dataset_mode euvp --batch_size 1
# Metrics: UIQM
python metrics/uiqm_metrics.py  --results_root  results/NonIdealRAS_In-air/test_latest/images/
```

## Apply a pre-trained model
- Download pre-trained models from [Google Drive](https://drive.google.com/file/d/1JCbrKT53JyAqF_sQFVecfkOxj86fsoZD/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1-tRIFz1Ju4ZHpzSjSmgvrQ) (access code: 70i0), and put `pre-trained/*` in the directory `checkpoints/`. Run test command to obtain the results.



<!-- 
# Bibtex
If you use this code for your research, please cite our papers.

```
@InProceedings{Guo_2021_ICCV,
    author    = {Guo, Zonghui and Guo, Dongsheng and Zheng, Haiyong and Gu, Zhaorui and Zheng, Bing and Dong, Junyu},
    title     = {Image Harmonization With Transformer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14870-14879}
}
``` -->

# Acknowledgement
For some of the data modules and model functions used in this source code, we need to acknowledge the repositories of [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [IntrinsicHarmony](https://github.com/zhenglab/IntrinsicHarmony). 
