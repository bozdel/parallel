for procs in {1..2}
do
	for size in {5..8}
	do
		mpirun.mpich -np $procs ./aloha2 $size $size >> time$procs.txt
		echo -e '\n' >> time.txt
	done
done