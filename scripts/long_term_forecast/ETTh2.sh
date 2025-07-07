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
embedding2=0
for pred_len in 96 192 
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id  ETTh2\
    --model  $model_name \
    --data ETTh2 \
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
    --cut1 $seq_len\
    --cut2 $pred_len \
    --dropout_total 0\
    --dropout_total2 0\
    --interaction 0\
    --linear 1\
    --seasonal 1\
    --trend 1\
    --timestamp 1\
    --embedding $embedding \
    --patch 0\
    --itr 1 --batch_size 128 --learning_rate 0.00001 >logs/LongForecasting_new/ETTh2_$model_name'_96_'$pred_len.log  
done


for pred_len in 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id  ETTh2\
    --model  $model_name \
    --data ETTh2 \
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
    --cut1 $seq_len\
    --cut2 $pred_len \
    --dropout_total 0\
    --dropout_total2 0\
    --interaction 0\
    --linear 1\
    --seasonal 1\
    --trend 1\
    --timestamp 1\
    --multiscale 0\
    --drop_initial 0\
    --embedding $embedding  $embedding2\
    --patch 0\
    --itr 1 --batch_size 128 --learning_rate 0.00001 >logs/LongForecasting_new/ETTh2_$model_name'_96_'$pred_len.log  
done