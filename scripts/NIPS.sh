export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.9 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Swan  --data_path dataset/NIPS_TS_Swan  --input_c 38 --output_c 38    --win_size 100  --use_revin
python main.py --anormly_ratio 0.9  --num_epochs 10     --batch_size 256   --mode test    --dataset NIPS_TS_Swan   --data_path dataset/NIPS_TS_Swan --input_c 38    --output_c 38       --win_size 100   --use_revin