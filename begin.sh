python run.py --gpu 0 --seed 0 --fold 10 --K1 5 --K2 7 --K3 7 --model 'BSGN' --d_output_root './result' --data 'camcan-movie' --d_subj_num 563 --seq_len 188 &
python run.py --gpu 0 --seed 0 --fold 10 --K1 5 --K2 7 --K3 7 --model 'BSGN' --d_output_root './result' --data 'camcan-rest' --d_subj_num 595 --seq_len 256  & 
python run.py --gpu 0 --seed 0 --fold 10 --K1 5 --K2 7 --K3 7 --model 'BSGN' --d_output_root './result' --data 'nki' --d_subj_num 1137 --seq_len 115  &

