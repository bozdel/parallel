#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include "dbg.h"
#include "misc.h"


#define FAST






void init_matrix(double *part, int part_size, int matr_size, int comm_size) {
	return;
}





int main(int argc, char *argv[]) {
	int Nx = 10;
	int Ny = 5;
	if (argc >= 3) {
		Nx = atoi(argv[1]);
		Ny = atoi(argv[2]);
	}
	int matr_size = Nx * Ny;
	double precision = 0.01;
	if (argc >= 4) {
		precision = atof(argv[3]);
	}

	int err_code;

	if ((err_code = MPI_Init(&argc, &argv)) != 0) {
		return err_code;
	}

	double *vec_b = (double*)malloc(matr_size * sizeof(double));
	double *vec_x = (double*)malloc(matr_size * sizeof(double));

	for (int i = 0; i < matr_size; i++) {
		vec_x[i] = 0;
		vec_b[i] = 0;
	}
	int dots_num = 6; // dots where temperature != 0
	for (int i = 0; i < dots_num; i++) {
		int ind = rand() % matr_size;
		vec_b[ind] = rand() % 100 - 50;
	}




	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	print_vec(vec_b, matr_size, comm_size, rank, NULL);

	
	// ------initing information------
	int cycle; // for checking cycles amount

	int part_size = part_size_by_rank(matr_size, comm_size, rank);

	// index of the first line of part in full matrix
	int my_shift = shift_by_rank(matr_size, comm_size, rank);
	

	int *displs = (int*)malloc(comm_size * sizeof(int));
	for (int proc_rank = 0; proc_rank < comm_size; proc_rank++) {
		displs[proc_rank] = shift_by_rank(matr_size, comm_size, proc_rank);
	}

	int *recvcounts = (int*)malloc(comm_size * sizeof(int));
	for (int proc_rank = 0; proc_rank< comm_size; proc_rank++) {
		recvcounts[proc_rank] = part_size_by_rank(matr_size, comm_size, proc_rank);
	}


	// ------initing parts------
	double *part = (double*)malloc(part_size * matr_size * sizeof(double));

	/*for (int i = 0; i < part_size; i++) {
		for (int j = 0; j < matr_size; j++) {
			part[i * matr_size + j] = 1;
		}
		part[i * matr_size + i + my_shift] += 1;
	}*/

	for (int i = 0; i < part_size; i++) {
		for (int j = 0; j < matr_size; j++) {
			part[i * matr_size + j] = 0;
		}
		part[i * matr_size + i + my_shift] = -4; // row (-4)
	}
	for (int i = -1; i < part_size; i++) {
		if ((my_shift + 1 + i) % Nx != 0) {
			if (0 <= my_shift + 1 + i && my_shift + 1 + i < matr_size) {
				part[i * matr_size + my_shift + 1 + i] = 1; // upper (1) row
			}
			if (0 <= i + 1 && i + 1 < part_size) {
				part[(i + 1) * matr_size + my_shift + i] = 1; // lower (1) row
			}
		}
	}

	
	print_matr(part, matr_size, part_size, comm_size, rank);

	// ||b||^2
	double norm2_b = scalar_mul(vec_b, vec_b, matr_size);


#ifdef FAST
	double *Ax = (double*)malloc(matr_size * sizeof(double));
	double *vec_y = (double*)malloc(matr_size * sizeof(double));
	double *Ay = (double*)malloc(matr_size * sizeof(double));


	matr_mul(part, part_size, vec_x, matr_size, my_shift, recvcounts, displs, Ax);
	sub(Ax, vec_b, matr_size, vec_y);
	for (cycle = 0 ; !check(vec_y, norm2_b, matr_size, precision); cycle++) {
		matr_mul(part, part_size, vec_y, matr_size, my_shift, recvcounts, displs, Ay);

		double yAy = scalar_mul(vec_y, Ay, matr_size);
		double AyAy = scalar_mul(Ay, Ay, matr_size);
		double tou = yAy / AyAy;

		subk(vec_x, tou, vec_y, matr_size, vec_x); // new x

		subk(Ax, tou, Ay, matr_size, Ax); // new Ax
		sub(Ax, vec_b, matr_size, vec_y); // new y
	}

#else
	double *vec_y = (double*)malloc(matr_size * sizeof(double));
	double *Ay = (double*)malloc(matr_size * sizeof(double));

	print_vec(vec_b, matr_size, comm_size, rank, "b");
	print_vec(vec_x, matr_size, comm_size, rank, "x");

	matr_mul(part, part_size, vec_x, matr_size, my_shift, recvcounts, displs, vec_y);

	print_vec(vec_y, matr_size, comm_size, rank, "Ax");	

	sub(vec_y, vec_b, matr_size, vec_y);

	print_vec(vec_y, matr_size, comm_size, rank, "Ax - b (y)");

	printf("%d prec: %d\n", rank, check(vec_y, norm2_b, matr_size, precision));
	MPI_Barrier(MPI_COMM_WORLD);
	for (cycle = 0 ; !check(vec_y, norm2_b, matr_size, precision); cycle++) {
		matr_mul(part, part_size, vec_y, matr_size, my_shift, recvcounts, displs, Ay);

		print_vec(Ay, matr_size, comm_size, rank, "Ay");

		sleep(1);

		double yAy = scalar_mul(vec_y, Ay, matr_size);
		double AyAy = scalar_mul(Ay, Ay, matr_size);
		double tou = yAy / AyAy;

		subk(vec_x, tou, vec_y, matr_size, vec_x); // new x
		print_vec(vec_x, matr_size, comm_size, rank, "x");

		matr_mul(part, part_size, vec_x, matr_size, my_shift, recvcounts, displs, vec_y);
		print_vec(vec_y, matr_size, comm_size, rank, "Ax");
		sub(vec_y, vec_b, matr_size, vec_y); // new y
		print_vec(vec_y, matr_size, comm_size, rank, "Ax - b");
	}

#endif

	if (rank == 0) {
		printf("cycles: %d\n", cycle);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	print_vec(vec_x, matr_size, comm_size, rank, NULL);
	print_vec(vec_y, matr_size, comm_size, rank, "y");


	MPI_Finalize();
	return 0;
}

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