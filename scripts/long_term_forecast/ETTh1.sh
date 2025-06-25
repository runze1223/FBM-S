export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

# seq_len=96
# e_layers=2
# down_sampling_layers=3
# down_sampling_window=2
# learning_rate=0.01
# d_model=16
# d_ff=32
# train_epochs=10
# patience=10
# batch_size=16



seq_len=336
model_name=FBM-Super
embedding=3


for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id  ETTh1\
    --model  $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7\
    --patch_num 14\
    --des 'Exp' \
    --train_epochs 100\
    --beta 1\
    --patience 3\
    --d_model2 128\
    --self_backbone 'MLP'\
    --cut1 $seq_len\
    --cut2 $pred_len \
    --dropout_total 0.5\
    --dropout_total2 0.5\
    --interaction 0\
    --linear 1\
    --seasonal 1\
    --trend 1\
    --timestamp 1\
    --multiscale 0\
    --drop_initial 0\
    --embedding $embedding \
    --patch 0\
    --itr 1 --batch_size 128 --learning_rate 0.00002 >logs/LongForecasting_new/ETTh1_$model_name'_96_'$pred_len.log  
done





















# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path  ./dataset/\
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 96 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'192 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 192 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 336 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 720 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window
