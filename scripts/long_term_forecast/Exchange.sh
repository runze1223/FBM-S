#export CUDA_VISIBLE_DEVICES=0



seq_len=336
model_name=FBM-Super


for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id  exchange\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8\
    --patch_num 14\
    --des 'Exp' \
    --train_epochs 100\
    --beta 1\
    --patience 10\
    --d_model2 128\
    --self_backbone 'MLP'\
    --cut1 24\
    --cut2 24 \
    --dropout_total 0\
    --dropout_total2 0.5\
    --interaction 0\
    --linear 1\
    --seasonal 1\
    --trend 1\
    --timestamp 0\
    --multiscale 0\
    --drop_initial 0\
    --dropout2 0.15\
    --d_model2 128 \
    --patch 0\
    --itr 1 --batch_size 128 --learning_rate 0.00002 >logs/LongForecasting_new/Exchange_$model_name'_336_'$pred_len.log  
done