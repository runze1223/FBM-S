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
