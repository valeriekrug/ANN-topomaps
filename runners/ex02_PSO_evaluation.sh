cd /project/ankrug/posthoc_topomaps/post-hoc-topomapping || return
python3 run_experiment.py -c configs/FAccT2022/ch5_experiments/ex02_PSO_evaluation_repNAP.json -o output -f -r 100 -t
python3 run_experiment.py -c configs/FAccT2022/ch5_experiments/ex02_PSO_evaluation_repTopo.json -o output -f -r 100 -t -rt
python3 run_experiment.py -c configs/FAccT2022/ch5_experiments/ex02b_CNN_repNAP.json -o output -f -r 100 -t
python3 run_experiment.py -c configs/FAccT2022/ch5_experiments/ex02b_CNN_repTopo.json -o output -f -r 100 -t -rt