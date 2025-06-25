#export CUDA_VISIBLE_DEVICES=0

# model_name=TimeMixer

# seq_len=96
# e_layers=3
# down_sampling_layers=3
# down_sampling_window=2
# learning_rate=0.01
# d_model=32
# d_ff=64
# batch_size=8



seq_len=96
model_name=FBM-Super
embedding=1
for pred_len in 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  Traffic\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len\
    --individual_embed 0\
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862\
    --e_layers 4 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 128 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 4\
    --des 'Exp' \
    --train_epochs 70\
    --beta 1\
    --patience 10\
    --d_model2 512\
    --dropout2 0.15\
    --self_backbone 'PatchTST'\
    --cut1 96\
    --cut2 $pred_len\
    --dropout_total 0\
    --dropout_total2 0\
    --interaction 1\
    --linear 0\
    --seasonal 1\
    --trend 1\
    --timestamp 1\
    --multiscale 0\
    --drop_initial 0\
    --hidden1 0\
    --hidden2 0\
    --drop_initial 0\
    --patch 1\
    --lradj 'ST'\
    --centralization 0\
    --embedding $embedding\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic_0.65_original6_$model_name'_96_'$pred_len.log  
done


# seq_len=336
# model_name=FBM-Super
# embedding=1
# for pred_len in 96 192 336 720
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path traffic.csv \
#     --model_id  Traffic\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len\
#     --individual_embed 0\
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862\
#     --e_layers 4 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 128 \
#     --dropout 0.2\
#     --fc_dropout 0.2\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 65\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --dropout2 0.15\
#     --self_backbone 'PatchTST'\
#     --cut1 336\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 1\
#     --multiscale 0\
#     --drop_initial 0\
#     --hidden1 0\
#     --hidden2 0\
#     --drop_initial 0\
#     --patch 1\
#     --lradj 'ST'\
#     --centralization 0\
#     --embedding $embedding\
#     --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic_0.65_original6_$model_name'_336_'$pred_len.log  
# done




# for pred_len in 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path traffic.csv \
#     --model_id  Traffic\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --dropout 0.2\
#     --fc_dropout 0.2\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 50\
#     --beta 1\
#     --patience 5\
#     --d_model2 512\
#     --self_backbone 'PatchTST'\
#     --cut1 12\
#     --cut2 4\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 1\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 1\
#     --multiscale 0\
#     --drop_initial 0\
#     --dropout 0.15\
#     --hidden1 256\
#     --hidden2 1440\
#     --drop_initial 0\
#     --patch 0\
#     --lradj 'ST'\
#     --centralization 1\
#     --embedding $embedding\
#     --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic_short_$model_name'_336_'$pred_len.log  
# done


# seq_len=336
# model_name=FBM-Super
# embedding=1
# for pred_len in 96 192 336 720  
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path traffic.csv \
#     --model_id  Traffic\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
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
#     --self_backbone 'PatchTST'\
#     --cut1 12\
#     --cut2 6\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 1\
#     --multiscale 0\
#     --drop_initial 0\
#     --dropout 0.15\
#     --hidden1 256\
#     --hidden2 1440\
#     --drop_initial 0\
#     --lradj 'ST'\
#     --centralization 1\
#     --embedding $embedding\
#     --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic_short_$model_name'_336_'$pred_len.log  
# done


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id Traffic_$seq_len'_'96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 96 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id Traffic_$seq_len'_'192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 192 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id Traffic_$seq_len'_'336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 336 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id Traffic_$seq_len'_'720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 720 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window