cd ../src
for i in `seq 0 10`; 
do
    python3 run.py --algorithm="dqn" --world="water" --num_times=1 --map=$i &&
    python3 run.py --algorithm="hrl" --world="water" --num_times=1 --map=$i &&
    python3 run.py --algorithm="hrl-rm" --world="water" --num_times=1 --map=$i &&
    python3 run.py --algorithm="qrm" --world="water" --num_times=1 --map=$i
done