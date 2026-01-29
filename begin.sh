# python run.py --gpu 0 --seed 0 --fold 10 --K1 9 --K2 9 --K3 3 --K4 5 --num_layers 5 --model 'BSGN' --d_output_root './result' --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
# python run.py --gpu 0 --seed 0 --fold 10 --K1 9 --K2 9 --K3 3 --K4 5 --num_layers 5 --model 'BSGN' --d_output_root './result' --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
# python run.py --gpu 0 --seed 0 --fold 10 --K1 9 --K2 9 --K3 3 --K4 5 --num_layers 5 --model 'BSGN' --d_output_root './result' --data 'nki' --d_subj_num 1137 --seq_len 115  &



# ps -ef | grep defunct | more


# "SEED=0 F=10 K1=9 K2=9 K3=3 K4=5 Layer=5 SAVE='./result_1'"
# "SEED=0 F=10 K1=5 K2=3 K3=9 K4=3 Layer=5 SAVE='./0_main_result'"
# "SEED=0 F=10 K1=9 K2=9 K3=3 K4=5 Layer=2 SAVE='./result_3'"

# # 定义参数组
PARAMS_LIST=(
    # "SEED=0 F=10 K1=5 K2=7 K3=7 SAVE='./result/0_main_result'"
    "SEED=0 F=10 K1=5 K2=7 K3=7 SAVE='./result/1_intra_state_ablation'"
)
# 定义公共执行逻辑
run_experiments() {
    # python run.py --gpu 0 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'BSGN' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    # python run.py --gpu 1 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation1' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    # python run.py --gpu 2 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation2' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation3' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    # python run.py --gpu 0 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove1' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    # python run.py --gpu 1 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove2' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    # python run.py --gpu 2 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove3' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Base1' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    python run.py --gpu 0 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Base2' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
    # python run.py --gpu 1 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove6' --d_output_root $SAVE --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &

    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'BSGN' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    # python run.py --gpu 1 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation1' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    # python run.py --gpu 0 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation2' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation3' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove1' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    # python run.py --gpu 0 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove2' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    # python run.py --gpu 1 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove3' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    python run.py --gpu 1 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Base1' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    python run.py --gpu 2 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Base2' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove6' --d_output_root $SAVE --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 

    # python run.py --gpu 1 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'BSGN' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation1' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    # python run.py --gpu 2 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation2' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Ablation3' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    # python run.py --gpu 0 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove1' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    # python run.py --gpu 2 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove2' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove3' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    python run.py --gpu 1 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Base1' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    python run.py --gpu 2 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Base2' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    # python run.py --gpu 3 --seed $SEED --fold $F --K1 $K1 --K2 $K2 --K3 $K3  --model 'Remove6' --d_output_root $SAVE --data 'nki' --d_subj_num 1137 --seq_len 115  &
    
    wait
}

# 遍历参数组并执行公共逻辑（并行）
# for PARAMS in "${PARAMS_LIST[@]}"; do
#     eval $PARAMS  # 将参数解析为变量
#     run_experiments &  # 每组参数以后台任务运行
# done

counter=0
for PARAMS in "${PARAMS_LIST[@]}"; do
    eval $PARAMS  
    run_experiments &  
    ((counter++))
    # 每x个任务后等待这x个任务完成
    if (( counter % 1 == 0 )); then
         wait  
    fi
done

# 等待所有参数组的任务完成
wait
