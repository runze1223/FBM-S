#export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ShortForecasting_new" ]; then
    mkdir ./logs/ShortForecasting_new
fi



# model_name=FBM-Super

# e_layers=4
# down_sampling_layers=1
# down_sampling_window=2
# learning_rate=0.0001
# d_model=32
# d_ff=32
# batch_size=64

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Monthly' \
#   --model_id m4_Monthly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 128 \
#   --d_model $d_model \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --train_epochs 150 \
#   --patience 20 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --loss 'SMAPE'\
#   --self_backbone 'MLP'\
#   --cut1 24\
#   --cut2 24\
#   --dropout_total 0\
#   --dropout_total2 0\
#   --interaction 0\
#   --linear 0\
#   --seasonal 1\
#   --trend 1\
#   --timestamp 0\
#   --dropout 0.2\
#   --multiscale 0\
#   --hidden1 1440\
#   --hidden2 1440\ 


# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Yearly' \
#   --model_id m4_Yearly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 128 \
#   --d_model $d_model \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --train_epochs 150 \
#   --patience 20 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --loss 'SMAPE'\
#   --self_backbone 'MLP'\
#   --cut1 24\
#   --cut2 24\
#   --dropout_total 0\
#   --dropout_total2 0\
#   --interaction 0\
#   --linear 0\
#   --seasonal 1\
#   --trend 1\
#   --timestamp 0\
#   --dropout 0.2\
#   --multiscale 0\
#   --hidden1 1440\
#   --hidden2 1440\ 

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Quarterly' \
#   --model_id m4_Quarterly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 128 \
#   --d_model $d_model \
#   --d_ff 64 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --train_epochs 150 \
#   --patience 20 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --loss 'SMAPE'\
#   --self_backbone 'MLP'\
#   --cut1 24\
#   --cut2 24\
#   --dropout_total 0\
#   --dropout_total2 0\
#   --interaction 0\
#   --linear 0\
#   --seasonal 1\
#   --trend 1\
#   --timestamp 0\
#   --dropout 0.2\
#   --multiscale 0\
#   --hidden1 1440\
#   --hidden2 1440\ 


# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Daily' \
#   --model_id m4_Daily \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 128 \
#   --d_model $d_model \
#   --d_ff 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --train_epochs 150 \
#   --patience 20 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --loss 'SMAPE'\
#   --self_backbone 'MLP'\
#   --cut1 24\
#   --cut2 24\
#   --dropout_total 0\
#   --dropout_total2 0\
#   --interaction 0\
#   --linear 0\
#   --seasonal 1\
#   --trend 1\
#   --timestamp 0\
#   --dropout 0.2\
#   --multiscale 0\
#   --hidden1 1440\
#   --hidden2 1440\ 


# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Weekly' \
#   --model_id m4_Weekly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 128 \
#   --d_model $d_model \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --train_epochs 150 \
#   --patience 20 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --loss 'SMAPE'\
#   --self_backbone 'MLP'\
#   --cut1 24\
#   --cut2 24\
#   --dropout_total 0\
#   --dropout_total2 0\
#   --interaction 0\
#   --linear 0\
#   --seasonal 1\
#   --trend 1\
#   --timestamp 0\
#   --dropout 0.2\
#   --multiscale 0\
#   --hidden1 1440\
#   --hidden2 1440\ 



# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Hourly' \
#   --model_id m4_Hourly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 128 \
#   --d_model $d_model \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --train_epochs 150 \
#   --patience 20 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --loss 'SMAPE'\
#   --self_backbone 'MLP'\
#   --cut1 24\
#   --cut2 24\
#   --dropout_total 0\
#   --dropout_total2 0\
#   --interaction 0\
#   --linear 0\
#   --seasonal 1\
#   --trend 1\
#   --timestamp 0\
#   --dropout 0.2\
#   --multiscale 0\
#   --hidden1 1440\
#   --hidden2 1440\  >logs/ShortForecasting_new/M4__$model_name'_summary_'.log  

######HERE

seq_len=336
model_name=FBM-Super

# for pred_len in 12 24 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS07.npz \
#     --model_id  PEMS07\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 883 \
#     --dec_in 883 \
#     --c_out 883\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS07_patch_cen_cut2max_$model_name'_336_'$pred_len.log  
# done




# for pred_len in 12 24 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS04.npz \
#     --model_id  PEMS04\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS04_patch_cen_cut2max_$model_name'_336_'$pred_len.log  
# done


# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 1440\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 0\
#     --centralization 1\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_patch_cen_cut2max2_$model_name'_96_'$pred_len.log  
# done


########working######################
# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 1\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_central+activation_$model_name'_96_'$pred_len.log  
# done



#%%


# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 256 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'PatchTST'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_efficiency256_$model_name'_96_'$pred_len.log  
# done







# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 1\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_noactivation+central_$model_name'_96_'$pred_len.log  
# done



# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 0\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 0\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_x1_$model_name'_96_'$pred_len.log  
# done


###here###########
# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 1\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_newcentral2_$model_name'_96_'$pred_len.log  
# done


# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 2\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 0\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_x1+d1+d2_$model_name'_96_'$pred_len.log  
# done




# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 336\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_cut96_$model_name'_96_'$pred_len.log  
# done



# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 336\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 0\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_nointer_$model_name'_96_'$pred_len.log  
# done

# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'PatchTST'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_PatchTST_$model_name'_96_'$pred_len.log  
# done





# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 1440\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 0\
#     --centralization 1\
#     --channel_mask 1\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_patch_cen_cut2max_mask$model_name'_96_'$pred_len.log  
# done


# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 48\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 1440\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 0\
#     --centralization 1\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_patch_cen_cut2max_48_$model_name'_96_'$pred_len.log  
# done

# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 96\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 1440\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 0\
#     --centralization 1\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_patch_cen_cut2max_96_$model_name'_96_'$pred_len.log  
# done




# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 $seq_len\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 1440\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 0\
#     --centralization 1\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_patch_cen_cut2max_max_$model_name'_96_'$pred_len.log  
# done


############visualize##########################


# for pred_len in 12 24 48
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_$model_name'_336_'$pred_len.log  
# done


# model_name=PatchTST

# for pred_len in 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 0\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 256\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_efficiency_$model_name'_336_'$pred_len.log  
# done








#%%%


# model_name=PatchTST

# for pred_len in 96 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --patch_len 16\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_visulize_$model_name'_336_'$pred_len.log  
# done


# model_name=iTransformer

# for pred_len in 96 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 512 \
#     --d_ff 512 \
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_visulize_$model_name'_336_'$pred_len.log  
# done


# model_name=TimeMixer
# seq_len=336
# down_sampling_layers=1
# down_sampling_window=2
# learning_rate=0.001
# d_model=128
# d_ff=256
# batch_size=64
# train_epochs=50
# patience=10

# for pred_len in 96 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 0 \
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id PEMS08 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --e_layers 5 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170 \
#     --des 'Exp' \
#     --itr 1 \
#     --use_norm 0 \
#     --channel_independence 0 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --batch_size 32 \
#     --learning_rate $learning_rate \
#     --train_epochs 10 \
#     --patience $patience \
#     --down_sampling_layers $down_sampling_layers \
#     --down_sampling_method avg \
#     --lradj 'TST'\
#     --down_sampling_window $down_sampling_window >logs/ShortForecasting_new/PEMS08_visulize_$model_name'_336_'$pred_len.log 
# done






# seq_len=336
# model_name=FBM-NL

# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS04.npz \
#     --model_id  PEMS04\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 512\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS04_$model_name'_96_'$pred_len.log  
# done


# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 512\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08_$model_name'_96_'$pred_len.log  
# done


# for pred_len in 12 24 48 96
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS03.npz \
#     --model_id  PEMS03\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 512\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS03_$model_name'_96_'$pred_len.log  
# done


# for pred_len in 12 24 48 96 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS07.npz \
#     --model_id  PEMS07\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 883 \
#     --dec_in 883 \
#     --c_out 883\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 512\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS07_$model_name'_96_'$pred_len.log  
# done




# for pred_len in 24 48 96  
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS07.npz \
#     --model_id  PEMS07\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307\
#     --e_layers 3 \
#     --n_heads 16 \
#     --des 'Exp' \
#     --train_epochs 100\
#     --embedding $embedding2  \
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --dropout 0.15\
#     --hidden1 1440\
#     --hidden2 1440\
#     --drop_initial 1\
#     --itr 1 --batch_size 32 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS07__$model_name'_96_'$pred_len.log  
# done





# seq_len=336
# model_name=FBM-Super

# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS04.npz \
#     --model_id  PEMS04\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 1440\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 1\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.002  >logs/ShortForecasting_new/PEMS04_patch_48_2_leaningrate0.002_$model_name'_96_'$pred_len.log  
# done





# seq_len=336
# model_name=FBM-Super

# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS04.npz \
#     --model_id  PEMS04\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 12\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 1440\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 0\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.001  >logs/ShortForecasting_new/PEMS04_nopatch_channelmask2_$model_name'_96_'$pred_len.log  
# done


# seq_len=336
# model_name=FBM-Super

# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS04.npz \
#     --model_id  PEMS04\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --fc_dropout 0.15\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 12\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --hidden1 1440\
#     --hidden2 1440\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --drop_initial 1\
#     --patch 0\
#     --centralization 0\
#     --channel_mask 0\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.001  >logs/ShortForecasting_new/PEMS04_nopatch_channelmask2_$model_name'_96_'$pred_len.log  
# done





# seq_len=336
# model_name=FBM-Super

# for pred_len in 48 
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS04.npz \
#     --model_id  PEMS04\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 512 \
#     --d_ff 256 \
#     --dropout 0.2\
#     --fc_dropout 0.2\
#     --head_dropout 0\
#     --patch_num 14\
#     --des 'Exp' \
#     --train_epochs 100\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'PatchTST'\
#     --cut1 24\
#     --cut2 12\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --dropout 0.15\
#     --dropout2 0.15\
#     --hidden1 128\
#     --hidden2 1440\
#     --drop_initial 1\
#     --patch 0\
#     --centralization 0\
#     --channel_mask 1\
#     --lradj 'TST'\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS04_channel_mask_$model_name'_96_'$pred_len.log  
# done




# for pred_len in  24 48 96  
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id  PEMS08\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170\
#     --e_layers 3 \
#     --n_heads 16 \
#     --des 'Exp' \
#     --train_epochs 100\
#     --embedding $embedding2  \
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 $pred_len\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --dropout 0.15\
#     --hidden1 1440\
#     --hidden2 1440\
#     --drop_initial 1\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS08__$model_name'_96_'$pred_len.log  
# done
# seq_len=336
# model_name=FBM-Super
# # embedding=0
# # embedding2=3

# for pred_len in 24 48 96  
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1\
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS03.npz \
#     --model_id  PEMS03\
#     --model  $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len\
#     --pred_len $pred_len \
#     --individual_embed 0\
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358\
#     --e_layers 3 \
#     --n_heads 16 \
#     --des 'Exp' \
#     --train_epochs 30\
#     --beta 1\
#     --patience 10\
#     --d_model2 512\
#     --self_backbone 'MLP'\
#     --cut1 24\
#     --cut2 12\
#     --dropout_total 0\
#     --dropout_total2 0\
#     --interaction 1\
#     --linear 0\
#     --seasonal 1\
#     --trend 1\
#     --timestamp 0\
#     --multiscale 1\
#     --dropout 0.15\
#     --hidden1 1440\
#     --hidden2 1440\
#     --drop_initial 1\
#     --itr 1 --batch_size 64 --learning_rate 0.0005  >logs/ShortForecasting_new/PEMS03_0.65_$model_name'_96_'$pred_len.log  
# done



##########################################################################################################################

# model_name=TimeMixer
# seq_len=336
# pred_len=48
# down_sampling_layers=1
# down_sampling_window=2
# learning_rate=0.003
# d_model=128
# d_ff=256
# batch_size=64
# train_epochs=50
# patience=10

# python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/PEMS/ \
#  --data_path PEMS03.npz \
#  --model_id PEMS03 \
#  --model $model_name \
#  --data PEMS \
#  --features M \
#  --seq_len $seq_len \
#  --label_len 0 \
#  --pred_len $pred_len \
#  --e_layers 5 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 358 \
#  --dec_in 358 \
#  --c_out 358 \
#  --des 'Exp' \
#  --itr 1 \
#  --use_norm 0 \
#   --channel_independence 0 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size 32 \
#  --learning_rate $learning_rate \
#  --train_epochs $train_epochs \
#  --patience $patience \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window >logs/ShortForecasting_new/PEMS03_look_$model_name'_96_'$pred_len.log  



# python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/PEMS/ \
#  --data_path PEMS04.npz \
#  --model_id PEMS04 \
#  --model $model_name \
#  --data PEMS \
#  --features M \
#  --seq_len $seq_len \
#  --label_len 0 \
#  --pred_len $pred_len \
#  --e_layers 5 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 307 \
#  --dec_in 307 \
#  --c_out 307 \
#  --des 'Exp' \
#  --itr 1 \
#  --use_norm 0 \
#  --channel_independence 0 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size 32 \
#  --learning_rate $learning_rate \
#  --train_epochs $train_epochs \
#  --patience $patience \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window  >logs/ShortForecasting_new/PEMS04_0.65_$model_name'_336_'$pred_len.log  


# python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/PEMS/ \
#  --data_path PEMS07.npz \
#  --model_id PEMS07 \
#  --model $model_name \
#  --data PEMS \
#  --features M \
#  --seq_len $seq_len \
#  --label_len 0 \
#  --pred_len $pred_len \
#  --e_layers 5 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 883 \
#  --dec_in 883 \
#  --c_out 883 \
#  --des 'Exp' \
#  --itr 1 \
#  --use_norm 0 \
#  --channel_independence 0 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size 32 \
#  --learning_rate $learning_rate \
#  --train_epochs $train_epochs \
#  --patience $patience \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window

# model_name=TimeMixer
# seq_len=336
# pred_len=96
# down_sampling_layers=1
# down_sampling_window=2
# learning_rate=0.003
# d_model=128
# d_ff=256
# batch_size=64
# train_epochs=50
# patience=10


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS08.npz \
#   --model_id PEMS08 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len $pred_len \
#   --e_layers 5 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 170 \
#   --dec_in 170 \
#   --c_out 170 \
#   --des 'Exp' \
#   --itr 1 \
#   --use_norm 0 \
#   --channel_independence 0 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 32 \
#   --learning_rate $learning_rate \
#   --train_epochs 10 \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window >logs/ShortForecasting_new/PEMS08_visulize_$model_name'_336_'$pred_len.log  
