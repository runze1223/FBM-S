export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ShortForecasting_new" ]; then
    mkdir ./logs/ShortForecasting_new
fi

model_name=FBM-S
learning_rate=0.0001
batch_size=64

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size\
  --des 'Exp' \
  --learning_rate $learning_rate \
  --train_epochs 150 \
  --patience 20 \
  --loss 'SMAPE'\
  --dropout 0.2\
  --hidden1 1440\
  --patch 0\
  --lradj 'TST' \


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size\
  --des 'Exp' \
  --learning_rate $learning_rate \
  --train_epochs 150 \
  --patience 20 \
  --loss 'SMAPE'\
  --dropout 0.2\
  --hidden1 1440\
  --patch 0\
  --lradj 'TST' \

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size\
  --des 'Exp' \
  --learning_rate $learning_rate \
  --train_epochs 150 \
  --patience 20 \
  --loss 'SMAPE'\
  --dropout 0.2\
  --hidden1 1440\
  --patch 0\
  --lradj 'TST' \

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size\
  --des 'Exp' \
  --learning_rate $learning_rate \
  --train_epochs 150 \
  --patience 20 \
  --loss 'SMAPE'\
  --dropout 0.2\
  --hidden1 1440\
  --patch 0\
  --lradj 'TST' \

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size\
  --des 'Exp' \
  --learning_rate $learning_rate \
  --train_epochs 150 \
  --patience 20 \
  --loss 'SMAPE'\
  --dropout 0.2\
  --hidden1 1440\
  --patch 0\
  --lradj 'TST' \

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size\
  --des 'Exp' \
  --learning_rate $learning_rate \
  --train_epochs 150 \
  --patience 20 \
  --loss 'SMAPE'\
  --dropout 0.2\
  --hidden1 1440\
  --patch 0\
  --lradj 'TST' \