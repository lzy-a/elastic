for predLen in 4  48 96
do

for model in FEDformer Autoformer Informer Transformer
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
 --pred_len $predLen \
 --e_layers 2 \
 --freq '15min' \
 --d_layers 1 \
 --train_epochs 8 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --d_model 512 \
 --itr 1

done

done