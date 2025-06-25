export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
batch_size=16




seq_len=336
model_name=FBM-Super

for pred_len in 336 
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTm2.csv \
    --model_id  ETTm2\
    --model  $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 7\
    --des 'Exp' \
    --train_epochs 100\
    --beta 1\
    --patience 5\
    --d_model2 128\
    --self_backbone 'MLP'\
    --cut1 12\
    --cut2 12\
    --dropout_total 0.1\
    --dropout_total2 0\
    --interaction 1\
    --linear 1\
    --seasonal 1\
    --trend 1\
    --timestamp 0\
    --multiscale 0\
    --dropout 0.2\
    --hidden1 128\
    --hidden2 720\
    --drop_initial 0\
    --itr 1 --batch_size 128 --learning_rate 0.0005  >logs/LongForecasting_new/ETTm2_$model_name'_96_'$pred_len.log  
done











# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path  ./dataset/ETT-small/\
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'96 \
#   --model $model_name \
#   --data ETTm2 \
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
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'192 \
#   --model $model_name \
#   --data ETTm2 \
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
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'336 \
#   --model $model_name \
#   --data ETTm2 \
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
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'720 \
#   --model $model_name \
#   --data ETTm2 \
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
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window
