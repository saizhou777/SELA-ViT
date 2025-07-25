now=$(date +"%Y%m%d_%H%M%S")

python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=990\
	main.py \
	--model dhvt_tiny_cifar_patch4 \
	--input-size 32 \
	--batch-size 256 \
	--warmup-epochs 5 \
   --lr 1e-3 \
   --num_workers 8 \
	--epochs 300 \
	--dist-eval \
   --data-set CIFAR \
	--data-path ./data/cifar-100-python \
	--output_dir ./output/cifar/$now

# python main.py --model dhvt_tiny_cifar_patch2 --input-size 32 --batch-size 64 --warmup-epochs 5 --lr 1e-3 --num_workers 8 --epochs 300 --data-set CIFAR --data-path E:\dataset\cifar-100-python --output_dir E:\run_in_this_computer\small_dataset202505121\DHVT-main\output
# python main.py --model deit_tiny_cifar_patch2 --input-size 32 --batch-size 64 --warmup-epochs 20 --lr 1e-3 --num_workers 4 --epochs 300 --data-set CIFAR --data-path E:\dataset\cifar-100-python --output_dir E:\run_in_this_computer\small_dataset202505121\DHVT-main\output
# python main.py --model sela_tiny_cifar_patch2 --input-size 32 --batch-size 128 --warmup-epochs 20 --lr 1e-3 --num_workers 4 --epochs 300 --data-set CIFAR --data-path E:\dataset\cifar-100-python --output_dir E:\run_in_this_computer\small_dataset202505121\DHVT-main\output
# python main.py --model sela_tiny_cifar_patch2 --input-size 32 --batch-size 128 --warmup-epochs 5 --lr 1e-3 --num_workers 4 --epochs 100 --data-set CIFAR --data-path E:\dataset\cifar-100-python --output_dir E:\run_in_this_computer\small_dataset202505121\DHVT-main\output

# python main.py --model dhvt_tiny_cifar_patch2 --input-size 32 --batch-size 64 --warmup-epochs 20 --lr 1e-3 --num_workers 4 --epochs 300 --data-set CIFAR --data-path E:\dataset\cifar-100-python --output_dir E:\run_in_this_computer\small_dataset202505121\DHVT-main\output