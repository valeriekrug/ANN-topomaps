cd /project/ankrug/posthoc_topomaps/post-hoc-topomapping || return
python3 run_experiment.py -c configs/FAccT2022/ch5_experiments/ex03a_unsupervised_NAP.json -o output -f -r 100
python3 run_experiment.py -c configs/FAccT2022/ch5_experiments/ex03b_unsupervised_random.json -o output -f -r 100
python3 run_experiment.py -c configs/FAccT2022/ch5_experiments/ex03c_unsupervised_balanced.json -o output -f -r 100