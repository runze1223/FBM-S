#export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi


seq_len=336
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
    --d_ff 128 \
    --dropout 0.2\
    --patch_num 14\
    --des 'Exp' \
    --train_epochs 70\
    --patience 10\
    --self_backbone 'PatchTST'\
    --cut1 $seq_len\
    --cut2 $pred_len\
    --interaction 1\
    --linear 0\
    --seasonal 1\
    --trend 1\
    --timestamp 1\
    --patch 1\
    --embedding $embedding\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic_$model_name'_96_'$pred_len.log  
done





