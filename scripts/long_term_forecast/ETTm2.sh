export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi


seq_len=336
model_name=FBM-S

for pred_len in 96 192 336 720
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
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --self_backbone 'MLP'\
    --dropout_total 0.2\
    --dropout_total2 0.2\
    --cut1 48\
    --cut2 48\
    --d_model2 128\
    --interaction 1\
    --linear 0\
    --seasonal 1\
    --trend 1\
    --patch 0\
    --hidden1 128\
    --hidden1 1440\
    --lradj 'TST'\
    --itr 1 --batch_size 128 --learning_rate  0.0004  >logs/LongForecasting_new/ETTm2_$model_name'_336_'$pred_len.log  
done






