cd ../src
python3 run.py --algorithm="dqn" --world="office" --num_times=30 &&
python3 run.py --algorithm="hrl" --world="office" --num_times=30 &&
python3 run.py --algorithm="hrl-rm" --world="office" --num_times=30 &&
python3 run.py --algorithm="qrm" --world="office" --num_times=30
