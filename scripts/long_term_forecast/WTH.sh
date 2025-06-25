#export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16
train_epochs=20
patience=10


seq_len=336
model_name=FBM-Super



# for pred_len in 96 192 336 720 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path weather.csv \
#     --model_id  weather\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 512 \
#     --d_ff 256 \
#     --dropout 0.2\
#     --fc_dropout 0.2\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 5\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 96\
#     --cut2 12\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --dropout 0.2\
#     --hidden1 256\
#     --hidden2 1440\
#     --drop_initial 0\
#     --itr 1 --batch_size 128 --learning_rate 0.00005  >logs/LongForecasting_new/weather_$model_name'_96_'$pred_len.log  
# done


for pred_len in 96 
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id  weather\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 256 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --des 'Exp' \
    --train_epochs 100\
    --beta 1\
    --patience 5\
    --d_model2 256\
    --self_backbone 'MLP'\
    --cut1 96\
    --cut2 12\
    --dropout_total 0\
    --dropout_total2 0\
    --interaction 1\
    --linear 0\
    --seasonal 1\
    --trend 1\
    --timestamp 0\
    --multiscale 1\
    --dropout 0.15\
    --hidden1 256\
    --hidden2 1440\
    --drop_initial 0\
    --centralization 1\
    --itr 1 --batch_size 128 --learning_rate 0.00005  >logs/LongForecasting_new/weather_activation_central_$model_name'_96_'$pred_len.log  
done


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 96 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 192 \
#   --e_layers $e_layers \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 336 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 720 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window