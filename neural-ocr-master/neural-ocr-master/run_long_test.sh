if [ -e "full_results.txt" ]
	then
		mv "full_results.txt" "full_results_"`date +"%H_%M_%S"`".txt"
fi

make clean
make all

bias=60
responseThreshold=40
learningRate=55

for training in `seq 100 100 1000`; do
	for testing in `seq 100 100 1000`; do
		for layers in `seq 1 2 20`; do
			#for bias in `seq 1 10 100`; do
			#	for responseThreshold in `seq 1 10 100`; do
			#		for learningRate in `seq 1 10 100`; do 
						for layerHeight in `seq 1 10 100`; do
							echo $training.$testing.$layers.$bias.$responseThreshold.$learningRate.$layerHeight
							./neural_read_base_case -t $training -T $testing -l $layers -b $bias -a $responseThreshold -r $learningRate -h $layerHeight
							./neural_read_single_core -t $training -T $testing -l $layers -b $bias -a $responseThreshold -r $learningRate -h $layerHeight
							./neural_read_multi_core_openmp -t $training -T $testing -l $layers -b $bias -a $responseThreshold -r $learningRate -h $layerHeight
							./neural_read_multi_core_pthreads -t $training -T $testing -l $layers -b $bias -a $responseThreshold -r $learningRate -h $layerHeight
						done
			#		done
			#	done
			#done
		done
	done
done

