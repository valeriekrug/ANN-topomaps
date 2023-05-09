cd /project/ankrug/posthoc_topomaps/post-hoc-topomapping || return
python3 run_config.py -c configs/FAccT2022/ch6_applications/appl01_testerror.json -o output -f
python3 run_config.py -c configs/FAccT2022/ch6_applications/appl01_trainerror.json -o output -f
python3 run_config.py -c configs/FAccT2022/ch6_applications/appl01_trainerror_confusion.json -o output -f