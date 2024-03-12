for model in FEDformer Autoformer Informer Transformer
do

for preLen in 2 4 8
do

python -u run.py \
 --is_training 1 \
 --root_path ./dataset/ \
 --data_path format_data.csv \
 --task_id kai \
 --data custom \
 --model $model \
 --target target\
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --train_epochs 10 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 1

done

done