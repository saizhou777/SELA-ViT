# SELA-ViT


## 🍀 1. Trianing Scripts

To train SELA-ViT-T on the ImageNet-1K dataset with two gpus, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=2 main_loss.py --cfg cfgs/deit_t.yaml --data-path /data2/saizhou777/ImageNet --output /data2/saizhou777/sela-main/classification/output
```

To train SELA-ViT-T on the small dataset with one gpu, please run:
```bash
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=29500 main.py --model deit_tiny_cifar_patch2 --input-size 32 --batch-size 128 --warmup-epochs 20 --lr 1e-3 --num_workers 4 --epochs 300 --dist-eval --data-set CIFAR --data-path /data2/saizhou777/data/cifar-100-python --output_dir /data2/saizhou777/sela-main/output
```

To train SELA-Track on the GOT-10k dataset with two gpus, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script ostrack --config selat_256_got10k_ep100 --save_dir ./output --mode multiple --nproc_per_node 2 --use_wandb 0
```


## ✨ 2. Inference Scripts

To eval SELA-ViT-T on the ImageNet-1K dataset on a single gpu, please identify the path of pretrained weight and run:
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=1 main_loss.py --cfg cfgs/deit_t.yaml --data-path /data2/saizhou777/ImageNet --output /data2/saizhou777/sela-main/classification/output/deit_tiny_patch16_224 --eval --resume /data2/saizhou777/sela-main/classification/output/deit_tiny_patch16_224/tiny/ckpt_epoch_300.pth
```
This will give
```bash
Acc@1 76.968 Acc@5 93.442
```

To eval SELA-DeiT-T on the ImageNet-1K dataset on a single gpu, please run:
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=1 main_loss.py --cfg cfgs/deit_t.yaml --data-path /data2/saizhou777/ImageNet --output /data2/saizhou777/sela-main/classification/output/deit_tiny_patch16_224 --eval --resume /data2/saizhou777/SLAB-main/classification/output/deit_tiny_patch16_224/sela_300/ckpt_epoch_300.pth
```
This will give
```bash
Acc@1 75.834 Acc@5 92.772
```


## 👏 3. Acknowledgement
This repository is built using [SLAB](https://github.com/xinghaochen/SLAB), [DHVT](https://github.com/ArieSeirack/DHVT), and [OSTrack](https://github.com/botaoye/OSTrack) repositories. We particularly appreciate their open-source efforts.


## 📖 4. Citation
If you find this repository helpful, please consider citing:
```bash
@Article{xxx
}
```
