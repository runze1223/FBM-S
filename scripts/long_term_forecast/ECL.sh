export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi




seq_len=336
model_name=FBM-S
embedding=1
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id  ECL\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321\
    --des 'Exp' \
    --train_epochs 100\
    --patience 5\
    --interaction 1\
    --linear 0\
    --seasonal 1\
    --trend 1\
    --timestamp 1\
    --lradj 'ST'\
    --patch 1\
    --embedding $embedding\
    --itr 1 --batch_size 16 --learning_rate 0.0005 >logs/LongForecasting_new/ECL_MLP_central$model_name'_96_'$pred_len.log  
done


# for pred_len in 96 192 336 720  
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path electricity.csv \
#     --model_id  ECL\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321\
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
#     --cut1 24\
#     --cut2 96\
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
#     --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/ECL_96_$model_name'_336_'$pred_len.log  
# done


# for pred_len in 96 192 336 720  
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path electricity.csv \
#     --model_id  ECL\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321\
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
#     --cut1 24\
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
#     --dropout 0.15\
#     --hidden1 256\
#     --hidden2 1440\
#     --drop_initial 0\
#     --lradj 'ST'\
#     --centralization 1\
#     --embedding $embedding\
#     --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/ECL_336_$model_name'_336_'$pred_len.log  
# done



# for pred_len in 96 192 336 720  
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path electricity.csv \
#     --model_id  ECL\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321\
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
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 24\
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
#     --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/ECL_MLP$model_name'_336_'$pred_len.log  
# done








# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_$seq_len'_'96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 96 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/Electricity__$model_name'_336_'96.log  




# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_$seq_len'_'192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 192 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/Electricity__$model_name'_336_'192.log  

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_$seq_len'_'336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 336 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/Electricity__$model_name'_336_'336.log  

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_$seq_len'_'720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 720 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/Electricity__$model_name'_336_'720.log 


  
# model_name=TimeMixer

# seq_len=336
# e_layers=3
# down_sampling_layers=3
# down_sampling_window=2
# learning_rate=0.01
# d_model=32
# d_ff=64
# batch_size=8


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
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/Traffic2__$model_name'_336_'96.log  

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
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/Traffic2__$model_name'_336_'192.log  

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
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/Traffic2__$model_name'_336_'336.log  

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
#   --down_sampling_window $down_sampling_window   >logs/LongForecasting_new/Traffic2__$model_name'_336_'720.log  



  
# model_name=TimeMixer

# seq_len=336
# e_layers=3
# down_sampling_layers=3
# down_sampling_window=2
# learning_rate=0.01
# d_model=16
# d_ff=32
# batch_size=16
# train_epochs=20
# patience=10

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path WTH.csv \
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
#   --enc_in 12 \
#   --dec_in 12 \
#   --c_out 12 \
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
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/WTH__$model_name'_336_'96.log  

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path WTH.csv \
#   --model_id weather_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 192 \
#   --e_layers $e_layers \
#   --factor 3 \
#   --enc_in 12 \
#   --dec_in 12 \
#   --c_out 12 \
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
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/WTH__$model_name'_336_'192.log  

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path WTH.csv \
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
#   --enc_in 12 \
#   --dec_in 12 \
#   --c_out 12 \
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
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/WTH__$model_name'_336_'336.log  

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path WTH.csv \
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
#   --enc_in 12 \
#   --dec_in 12 \
#   --c_out 12 \
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
#   --down_sampling_window $down_sampling_window  >logs/LongForecasting_new/WTH__$model_name'_336_'720.log  