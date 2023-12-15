dataset="HCPAge"
runs=1
device="cpu"
hidden=32
hidden_mlp=64
num_layers=3
epochs=50
echo_epoch=50
batch_size=16
early_stopping=50
lr=0.00001
weight_decay=0.0005
dropout=0.5
python main.py --dataset $dataset --runs $runs --device $device --hidden $hidden --hidden_mlp $hidden_mlp --num_layers $num_layers --epochs $epochs --echo_epoch $echo_epoch --batch_size $batch_size --early_stopping $early_stopping --lr $lr --weight_decay $weight_decay --dropout $dropout