cd - 
amountCheckpoints=$(find output_base_path/checkpoints_for_video -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "$amountCheckpoints"
for i in $(seq 0 $(($amountCheckpoints-1))); do python3 create_topomaps_from_checkpoints.py -i $i; done