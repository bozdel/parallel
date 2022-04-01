#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// ???
double scal_mul() {}

// shift - is way to determine where to put result of mul into dest (which part of matrix was multiplied)
void part_matr_mul(double *part_matr, int part_size, double *vec, int size, double *dest, int shift) {
	for (int i = 0; i < part_size; i++) {
		double *line = part_matr + i * size;
		double tmp_res = 0.0;
		for (int j = 0; j < size; j++) {
			tmp_res += line[j] * vec[j];
		}
		dest[shift + i] = tmp_res;
	}
}

double scalar_mul_allred(double *vec1, double *vec2, int size) {
	double sub_res = 0;
	for (int i = 0; i < size; i++) {
		sub_res += vec1[i] * vec2[i];
	}
	double res = 0;
	MPI_Allreduce(&sub_res, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return res;
}

double scalar_mul(double *vec1, double *vec2, int size) {
	double res = 0;
	for (int i = 0; i < size; i++) {
		res += vec1[i] * vec2[i];
	}
	return res;
}

void init_matrix(double *part, int part_size, int matr_size, int comm_size) {
	return;
}

void print_part(double *part, int matr_size, int part_size) {
	for (int i = 0; i < part_size; i++) {
		for (int j = 0; j < matr_size; j++) {
			printf("%.0f ", part[i * matr_size + j]);
		}
		printf("\n");
	}
}

void print_matr(double *part, int matr_size, int part_size, int comm_size, int rank) {
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < comm_size; i++) {
		if (rank == i) {
			print_part(part, matr_size, part_size);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void print_vec(double *vec, int vec_size, int comm_size, int rank) {
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < comm_size; i++) {
		if (rank == i) {
			printf("%d: ", rank);
			for (int i = 0; i < vec_size; i++) {
				printf("%.0f ", vec[i]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}	
}

void print_vecint(int *vec, int vec_size, int comm_size, int rank) {
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < comm_size; i++) {
		if (rank == i) {
			printf("%d: ", rank);
			for (int i = 0; i < vec_size; i++) {
				printf("%d ", vec[i]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}	
}

int part_size_by_rank(int matr_size, int comm_size, int rank) {
	int part_size = matr_size / comm_size;
	if (rank < (matr_size % comm_size)) return part_size + 1;
	return part_size;
}

int shift_by_rank(int matr_size, int comm_size, int rank) {
	int my_shift;
	if (rank <= (matr_size % comm_size)) { // first group of process with one additional line from distributed remain/rest
		return (matr_size / comm_size + 1) * rank;
	}
	else { // second group
		return (matr_size / comm_size + 1) * (matr_size % comm_size) + (matr_size / comm_size) * (rank - (matr_size % comm_size));
	}
}

int main(int argc, char *argv[]) {
	int matr_size = 100;
	if (argc >= 2) {
		matr_size = atoi(argv[1]);
	}

	int err_code;

	if ((err_code = MPI_Init(&argc, &argv)) != 0) {
		return err_code;
	}

	double *vec_b = (double*)malloc(matr_size * sizeof(double));
	double *vec_x = (double*)malloc(matr_size * sizeof(double));

	for (int i = 0; i < matr_size; i++) {
		vec_x[i] = 0;
		vec_b[i] = matr_size + 1;
	}




	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	

	int part_size = part_size_by_rank(matr_size, comm_size, rank);
	printf("my rank: %d, part size: %d\n", rank, part_size);


	MPI_Barrier(MPI_COMM_WORLD);

	// index of the first line of part in full matrix
	int my_shift = shift_by_rank(matr_size, comm_size, rank);
	printf("my rank: %d, my shift: %d\n", rank, my_shift);
	
	int *displs = (int*)malloc(comm_size * sizeof(int));
	for (int proc_rank = 0; proc_rank < comm_size; proc_rank++) {
		displs[proc_rank] = shift_by_rank(matr_size, comm_size, proc_rank);
	}


	int *recvcounts = (int*)malloc(comm_size * sizeof(int));
	for (int proc_rank = 0; proc_rank< comm_size; proc_rank++) {
		recvcounts[proc_rank] = part_size_by_rank(matr_size, comm_size, proc_rank);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	print_vecint(displs, comm_size, comm_size, rank);
	MPI_Barrier(MPI_COMM_WORLD);
	print_vecint(recvcounts, comm_size, comm_size, rank);

	// initing parts
	double *part = (double*)malloc(part_size * matr_size * sizeof(double));

	for (int i = 0; i < part_size; i++) {
		for (int j = 0; j < matr_size; j++) {
			part[i * matr_size + j] = 1;
		}
		part[i * matr_size + i + my_shift] += 1;
	}

	
	MPI_Barrier(MPI_COMM_WORLD);
	print_matr(part, matr_size, part_size, comm_size, rank);

	

	
	double *tmp = (double*)malloc(matr_size * sizeof(double));
	part_matr_mul(part, part_size, vec_b, matr_size, tmp, my_shift);
	print_vec(tmp, matr_size, comm_size, rank);



	double *tmp2 = (double*)malloc(matr_size * sizeof(double));
	MPI_Allgatherv(tmp + my_shift, part_size, MPI_DOUBLE, tmp2, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);



	MPI_Barrier(MPI_COMM_WORLD);
	print_vec(tmp2, matr_size, comm_size, rank);



	MPI_Finalize();
	return 0;
}

#if 1
int mainv1(int argc, char *argv[]) {
	int matr_size = 100;
	if (argc >= 2) {
		matr_size = atoi(argv[1]);
	}

	int err_code;

	if ((err_code = MPI_Init(&argc, &argv)) != 0) {
		return err_code;
	}

	double *vec_b = (double*)malloc(matr_size * sizeof(double));
	double *vec_x = (double*)malloc(matr_size * sizeof(double));

	for (int i = 0; i < matr_size; i++) {
		vec_x[i] = 0;
		vec_b[i] = matr_size + 1;
	}



	// double arr[10] = { 0 };

	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int part_size = matr_size / comm_size;
	int lpart_size = part_size + matr_size % comm_size; // last part size
	

	double *part = (double*)malloc(lpart_size * matr_size * sizeof(double));

	if (rank != comm_size - 1) { // not last proc
		for (int i = 0; i < part_size; i++) {
			for (int j = 0; j < matr_size; j++) {
				part[i * matr_size + j] = 1;
			}
			part[i * matr_size + i + part_size * rank] += 1;
		}
	}
	if (rank == comm_size - 1) {
		for (int i = 0; i < lpart_size; i++) {
			for (int j = 0; j < matr_size; j++) {
				part[i * matr_size + j] = 1;
			}
			part[i * matr_size + i + part_size * rank] += 1;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < comm_size; i++) {
		if (rank == i && rank != comm_size - 1) {
			print_part(part, matr_size, lpart_size);
		}
		if (rank == i && rank == comm_size - 1) {
			print_part(part, matr_size, lpart_size);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	double *tmp = (double*)malloc(matr_size * sizeof(double));
	if (rank != comm_size - 1) {
		part_matr_mul(part, part_size, vec_b, matr_size, tmp, part_size * rank);
	}
	else {
		part_matr_mul(part, lpart_size, vec_b, matr_size, tmp, part_size * rank);	
	}

	
	print_vec(tmp, matr_size, comm_size, rank);

	double *tmp2 = (double*)malloc(matr_size * sizeof(double));

	int *displs = (int*)malloc(comm_size * sizeof(int));
	for (int i = 0; i < comm_size; i++) {
		displs[i] = i * part_size;
	}
	int *recvcounts = (int*)malloc(comm_size * sizeof(int));
	for (int i = 0; i < comm_size; i++) {
		recvcounts[i] = part_size;
		if (i == comm_size - 1) {
			recvcounts[i] += matr_size % comm_size;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	print_vecint(displs, comm_size, comm_size, rank);
	MPI_Barrier(MPI_COMM_WORLD);
	print_vecint(recvcounts, comm_size, comm_size, rank);

	if (rank != comm_size - 1) {
		MPI_Allgatherv(tmp + part_size * rank, part_size, MPI_DOUBLE, tmp2, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
	}
	else {
		MPI_Allgatherv(tmp + part_size * rank, lpart_size, MPI_DOUBLE, tmp2, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
	}

	// MPI_Allgather(tmp + part_size * rank, part_size, MPI_DOUBLE, tmp2, part_size, MPI_DOUBLE, MPI_COMM_WORLD);

	
	MPI_Barrier(MPI_COMM_WORLD);
	print_vec(tmp2, matr_size, comm_size, rank);

	// init_matrix(part, part_size, matr_size, comm_size);

	/*arr[rank] = rank;

	printf("my rank is: %d\n", rank);

	for (int i = 0; i < 10; i++) {
		printf("%f ", arr[i]);
	}
	printf("\n");

	MPI_Barrier(MPI_COMM_WORLD);

	// gather(from where, where to put)
	// shoould it be called by one of processes or by each?
	// MPI_Allgather(x, ~ size / rank, MPI_DOUBLE, ~ size / rank, MPI_DOUBLE, MPI_COMM_WORLD);
	double *recvd = (double*)malloc(sizeof(double) * 10);
	MPI_Allgather(arr + rank, 1, MPI_DOUBLE, recvd, 1, MPI_DOUBLE, MPI_COMM_WORLD);

	printf("my rank is %d as i said\n", rank);
	for (int i = 0; i < 10; i++) {
		printf("%f ", recvd[i]);
	}
	printf("\n");*/



	MPI_Finalize();
	return 0;
}
#endif

#if 0
int main(int argc, char *argv[]) {
	int err_code;

	if ((err_code = MPI_Init(&argc, &argv)) != 0) {
		return err_code;
	}

	double arr[10] = { 0 };

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("my rank is: %d\n", rank);

	int var = 10;
	int buff = 0;

	if (rank == 0) {
		/*MPI_Send(&var, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		printf("data was sent: %d\n", var);*/
	}
	else if (rank == 1) {
		/*MPI_Recv(&buff, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("data was recieved: %d\n", buff);*/
	}

	// part_matr_mul(some_part_of_matrix, ~ size / rank, vector, dest_vector, ~ size / rank);

	// gather(from where, where to put)
	// shoould it be called by one of processes or by each?
	MPI_Allgather(x, ~ size / rank, MPI_DOUBLE, ~ size / rank, MPI_DOUBLE, MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}
#endif

#if 0
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
/**
 * @brief Illustrates how to use an allgather.
 * @details This application is meant to be run with 3 MPI processes. Every MPI
 * process begins with a value, then every MPI process collects the entirety of
 * the data gathered and prints them. It can be visualised as follows:
 *
 * +-----------+  +-----------+  +-----------+
 * | Process 0 |  | Process 1 |  | Process 2 |
 * +-+-------+-+  +-+-------+-+  +-+-------+-+
 *   | Value |      | Value |      | Value |
 *   |   0   |      |  100  |      |  200  |
 *   +-------+      +-------+      +-------+
 *       |________      |      ________|
 *                |     |     | 
 *             +-----+-----+-----+
 *             |  0  | 100 | 200 |
 *             +-----+-----+-----+
 *             |   Each process  |
 *             +-----------------+
 **/
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
 
    // Get number of processes and check that 3 processes are used
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 3)
    {
        printf("This application is meant to be run with 3 MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
    // Get my rank
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 
    // Define my value
    // int my_value = my_rank * 100;
    // printf("Process %d, my value = %d.\n", my_rank, my_value);
    int my_arr[3] = { 0 };
    my_arr[my_rank] = my_rank;
    printf("Process %d: %d %d %d\n", my_rank, my_arr[0], my_arr[1], my_arr[2]);
 
    int buffer[3];
    MPI_Allgather(my_arr + my_rank, 1, MPI_INT, my_arr, 1, MPI_INT, MPI_COMM_WORLD);
    printf("Values collected on process %d: %d, %d, %d.\n", my_rank, buffer[0], buffer[1], buffer[2]);
 
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}
#endif