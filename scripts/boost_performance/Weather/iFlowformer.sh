export mps_VISIBLE_DEVICES=2

model_name=iFlowformer
# model_name=Flowformer

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 128 \
    --train_epochs 3

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_192 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --batch_size 128 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_336 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --batch_size 128 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --batch_size 128 \
    --itr 1
