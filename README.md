# SELA-ViT


## 🍀 1. Trianing Scripts

To train SELA-ViT-T on the ImageNet-1K dataset with two gpus, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=2 main_loss.py --cfg cfgs/deit_t.yaml --data-path /data2/saizhou777/ImageNet --output /data2/saizhou777/sela-main/classification/output
```


## ✨ 2. Inference Scripts



## 👏 3. Acknowledgement
This repository is built using [SLAB](https://github.com/xinghaochen/SLAB), [DHVT](https://github.com/ArieSeirack/DHVT), and [OSTrack](https://github.com/botaoye/OSTrack) repositories. We particularly appreciate their open-source efforts.


## 📖 4. Citation
If you find this repository helpful, please consider citing:
```bash
@Article{xxx
}
```
