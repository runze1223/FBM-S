if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ShortForecasting_new" ]; then
    mkdir ./logs/ShortForecasting_new
fi



seq_len=336
model_name=FBM-L

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id  PEMS03\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS03__$model_name'_96_'$pred_len.log  
done


for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id  PEMS08\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08__$model_name'_96_'$pred_len.log  
done


for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id  PEMS04\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS04__$model_name'_96_'$pred_len.log  
done



for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id  PEMS07\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS07__$model_name'_96_'$pred_len.log  
done


seq_len=336
model_name=NLinear

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id  PEMS03\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.003  >logs/ShortForecasting_new/PEMS03__$model_name'_96_'$pred_len.log  
done


for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id  PEMS08\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.003  >logs/ShortForecasting_new/PEMS08__$model_name'_96_'$pred_len.log  
done


for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id  PEMS04\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.003  >logs/ShortForecasting_new/PEMS04__$model_name'_96_'$pred_len.log  
done



for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id  PEMS07\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.003  >logs/ShortForecasting_new/PEMS07__$model_name'_96_'$pred_len.log  
done



seq_len=336
model_name=FBM-NP

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id  PEMS03\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS03__$model_name'_96_'$pred_len.log  
done


for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id  PEMS08\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08__$model_name'_96_'$pred_len.log  
done


for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id  PEMS04\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS04__$model_name'_96_'$pred_len.log  
done



for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id  PEMS07\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --label_len 48 \
    --pred_len $pred_len \
    --individual_embed 0\
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_num 14\
    --decomposition 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS07__$model_name'_96_'$pred_len.log  
done