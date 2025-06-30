#export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi

seq_len=336
model_name=FBM-Super


for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id  weather\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21\
    --des 'Exp' \
    --train_epochs 100\
    --patience 5\
    --self_backbone 'MLP'\
    --cut1 96\
    --cut2 12\
    --dropout2 0.2\
    --d_model2 256\
    --interaction 1\
    --linear 1\
    --seasonal 1\
    --trend 1\
    --multiscale 1\
    --centralization 1\
    --itr 1 --batch_size 128 --learning_rate 0.00005  >logs/LongForecasting_new/weather_activation_central_$model_name'_96_'$pred_len.log  
done


