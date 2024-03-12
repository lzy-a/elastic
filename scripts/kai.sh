for model in FEDformer Autoformer Informer Transformer
do

for labelLen in 24 48 96
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
 --label_len $labelLen \
 --pred_len 4 \
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