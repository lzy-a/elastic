for model in Informer Transformer
do

for labelLen in 24 96
do

python -u run.py \
 --is_training 0 \
 --do_predict  \
 --root_path ./dataset/ \
 --data_path predict_data.csv \
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