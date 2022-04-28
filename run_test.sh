for size in {3..5}
do
	for dots_num in {6..10}
	do
		mpirun.mpich -np 5 ./aloha $size $size 0.01 $dots_num >> ress
	# for size in {5..8}
	# do
	# 	mpirun.mpich -np $procs ./aloha2 $size $size >> time$procs.txt
	# 	echo -e '\n' >> time.txt
	# done
	done
done