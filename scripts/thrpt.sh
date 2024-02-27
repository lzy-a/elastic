for model in FEDformer Autoformer Informer Transformer
do

python -u run.py \
 --is_training 1 \
 --root_path ./dataset/ \
 --data_path throughput_dataset.csv \
 --task_id infer \
 --data custom \
 --model $model \
 --target throughput \
 --features S \
 --seq_len 96 \
 --label_len 12 \
 --pred_len 1 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3

done