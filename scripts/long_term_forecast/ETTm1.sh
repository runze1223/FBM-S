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
    --data_path ETTm1.csv \
    --model_id  ETTm1\
    --model  $model_name \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7\
    --des 'Exp' \
    --train_epochs 100\
    --patience 5\
    --d_model2 128\
    --self_backbone 'MLP'\
    --cut1 48\
    --cut2 48\
    --dropout_total 0.2\
    --dropout_total2 0.2\
    --interaction 1\
    --linear 0\
    --seasonal 1\
    --trend 1\
    --timestamp 0\
    --multiscale 2\
    --dropout 0.15\
    --hidden1 128\
    --hidden2 1440\
    --centralization 1\
    --patch 1\
    --itr 1 --batch_size 128 --learning_rate 0.00004 >logs/LongForecasting_new/ETTm1_$model_name'_96_'$pred_len.log  
done

