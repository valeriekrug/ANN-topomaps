cd -
echo $1 'is the path'
amountCheckpoints=$(find checkpoints_for_tp -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "$amountCheckpoints checkpoints found"
for i in $(seq 0 $(($amountCheckpoints-1))); do "C:\Users\chris\PycharmProjects\MasterThesisPython\venv\Scripts\python.exe" create_topomaps_from_checkpoints.py -c $1 -i $i; done