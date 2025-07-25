export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi



seq_len=336
model_name=FBM-S
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
    --patience 3\
    --d_model2 128\
    --self_backbone 'MLP'\
    --dropout_total 0.5\
    --interaction 0\
    --linear 1\
    --seasonal 1\
    --trend 1\
    --timestamp 1\
    --embedding $embedding \
    --patch 0\
    --itr 1 --batch_size 128 --learning_rate 0.00002 >logs/LongForecasting_new/ETTh1_$model_name'_96_'$pred_len.log  
done


















