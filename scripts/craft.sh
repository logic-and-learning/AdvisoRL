cd ../src
for i in `seq 0 10`; 
do
	python3 run.py --algorithm="dqn" --world="craft" --num_times=3 --map=$i &&
	python3 run.py --algorithm="hrl" --world="craft" --num_times=3 --map=$i &&
	python3 run.py --algorithm="hrl-rm" --world="craft" --num_times=3 --map=$i &&
	python3 run.py --algorithm="qrm" --world="craft" --num_times=3 --map=$i
done