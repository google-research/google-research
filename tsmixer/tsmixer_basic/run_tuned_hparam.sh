set -x
data=$1
seq_len=512
pred_len=96

if [ $data = "ETTm2" ]
then
    python run.py --model TSMixerRevIN --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate 0.001 --n_block 2 --dropout 0.9 --ff_dim 64
elif [ $data = "weather" ]
then
    python run.py --model TSMixerRevIN --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate 0.0001 --n_block 4 --dropout 0.3 --ff_dim 32
elif [ $data = "electricity" ]
then
    python run.py --model TSMixerRevIN --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate 0.0001 --n_block 4 --dropout 0.7 --ff_dim 64
elif [ $data = "traffic" ]
then
    python run.py --model TSMixerRevIN --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate 0.0001 --n_block 8 --dropout 0.7 --ff_dim 64
else
    echo "Unknown dataset"
fi
