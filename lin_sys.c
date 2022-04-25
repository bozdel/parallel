#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include "dbg.h"
#include "misc.h"


#define FAST






void init_matrix_v1(double *part, int part_size, int matr_size, int my_shift) {
	for (int i = 0; i < part_size; i++) {
		for (int j = 0; j < matr_size; j++) {
			part[i * matr_size + j] = 1;
		}
		part[i * matr_size + i + my_shift] += 1;
	}
}

void init_matrix_v3(double *part, int part_size, int matr_size, int my_shift, int Nx, int Ny) {
	for (int i = 0; i < part_size; i++) {
		for (int j = 0; j < matr_size; j++) {
			part[i * matr_size + j] = 0;
		}
		part[i * matr_size + i + my_shift] = -4; // row (-4)
	}
	for (int i = 0; i < part_size; i++) { // upper (1) row
		if (i * matr_size + i + 1 + my_shift < matr_size * matr_size) {
			part[i * matr_size + i + 1 + my_shift] = 1;
		}
	}
	for (int i = 0; i < part_size; i++) { // lower (1) row
		if (i * matr_size + i - 1 + my_shift > 0) {
			part[i * matr_size + i - 1 + my_shift] = 1;
		}
	}
	for (int i = 0; i < part_size; i++) { // upper zero shifted row
		if (i * matr_size + i + 1 + my_shift < matr_size * matr_size && (i + my_shift + 1) % Nx == 0) {
			part[i * matr_size + i + 1 + my_shift] = 0;
		}
	}
	for (int i = 0; i < part_size; i++) { // lower (1) row
		if (i * matr_size + i - 1 + my_shift > 0 && (i + my_shift) % Nx == 0) {
			part[i * matr_size + i - 1 + my_shift] = 0;
		}
	}
	for (int i = 0; i < part_size; i++) {
		if (i * matr_size + i + Nx + my_shift < matr_size * matr_size && (i + Nx + my_shift) < matr_size) {
			part[i * matr_size + i + Nx + my_shift] = 1;
		}
		if (i * matr_size + i - Nx + my_shift >= 0 && (i - Nx + my_shift) >= 0) {
			part[i * matr_size + i - Nx + my_shift] = 1;
		}
	}
}


// returns cycles amont
// vec_x - destination vector
int solve_fast(double *part, int part_size, double *vec_b, double *vec_x, int size, double precision, int *displs, int *recvcounts, int rank) {
	int cycle = 0;

	double norm2_b = scalar_mul(vec_b, vec_b, size);

	// something wrong with allocation ( -np 4 ./prog 3 7)
	double *Ax = (double*)malloc(size * sizeof(double));
	double *vec_y = (double*)malloc(size * sizeof(double));
	double *Ay = (double*)malloc(size * sizeof(double));
	// temporary vector for matr_mul func
	double *mm_tmp = (double*)malloc(part_size * sizeof(double));

	matr_mul(part, part_size, vec_x, size, recvcounts, displs, Ax, mm_tmp); // Ax
	sub(Ax, vec_b, size, vec_y); // y = Ax - b
	for (cycle = 0 ; !check(vec_y, norm2_b, size, precision); cycle++) { // check - (||Ax-b||/||b||)^2 < precision^2
		matr_mul(part, part_size, vec_y, size, recvcounts, displs, Ay, mm_tmp); // Ay

		double yAy = scalar_mul(vec_y, Ay, size);
		double AyAy = scalar_mul(Ay, Ay, size);
		double tou = yAy / AyAy;

		subk(vec_x, tou, vec_y, size, vec_x); // new x = x - ty

		subk(Ax, tou, Ay, size, Ax); // new Ax. A(x - ty) = Ax - tAy
		sub(Ax, vec_b, size, vec_y); // new y. y = Ax - b
	}

	return cycle;
}

// same signature as solve_fast
int solve(double *part, int part_size, double *vec_b, double *vec_x, int size, double precision, int *displs, int *recvcounts, int rank) {
	int cycle = 0;

	double norm2_b = scalar_mul(vec_b, vec_b, size);

	double *vec_y = (double*)malloc(size * sizeof(double));
	double *Ay = (double*)malloc(size * sizeof(double));
	// temporary vector for matr_mul func
	double *mm_tmp = (double*)malloc(part_size * sizeof(double));

	matr_mul(part, part_size, vec_x, size, recvcounts, displs, vec_y, mm_tmp);

	sub(vec_y, vec_b, size, vec_y);

	for (cycle = 0 ; !check(vec_y, norm2_b, size, precision); cycle++) {
		matr_mul(part, part_size, vec_y, size, recvcounts, displs, Ay, mm_tmp);

		double yAy = scalar_mul(vec_y, Ay, size);
		double AyAy = scalar_mul(Ay, Ay, size);
		double tou = yAy / AyAy;

		subk(vec_x, tou, vec_y, size, vec_x); // new x

		matr_mul(part, part_size, vec_x, size, recvcounts, displs, vec_y, mm_tmp);

		sub(vec_y, vec_b, size, vec_y); // new y
	}

	return cycle;
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
		printf("error\n");
		return err_code;
	}

	double *vec_b = (double*)malloc(matr_size * sizeof(double));
	double *vec_x = (double*)malloc(matr_size * sizeof(double));

	for (int i = 0; i < matr_size; i++) {
		vec_x[i] = 0;
		vec_b[i] = 0;
	}
	// 1-st variant
	/*for (int i = 0; i < matr_size; i++) {
		vec_b[i] = matr_size + 1;
		vec_x[i] = 0;
	}*/
	// 3-rd variant
	int dots_num = 6; // dots where temperature != 0
	for (int i = 0; i < dots_num; i++) {
		int ind = rand() % matr_size;
		vec_b[ind] = rand() % 100 - 50;
	}




	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	// print_vec(vec_b, matr_size, comm_size, rank, NULL);

	
	// ------initing information------

	int part_size = part_size_by_rank(matr_size, comm_size, rank);

	// can be raplaces by displs[rank_number]
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

	// 1-st variant
	
	// init_matrix_v1(part, part_size, matr_size, my_shift);

	// 3-rd variant

	init_matrix_v3(part, part_size, matr_size, my_shift, Nx, Ny);


	// print_matr(part, matr_size, part_size, comm_size, rank);


	int cycle = 0; // for checking cycles amount

	clock_t beg = clock();

#ifdef FAST
	cycle = solve_fast(part, part_size, vec_b, vec_x, matr_size, precision, displs, recvcounts, rank);
#else
	cycle = solve(part, part_size, vec_b, vec_x, matr_size, precision, displs, recvcounts, rank);
#endif

	clock_t end = clock();

	printf("%f\n", (float)(end - beg) / CLOCKS_PER_SEC);
	if (rank == 0) {
		printf("cycles: %d\n", cycle);
	}

	// MPI_Barrier(MPI_COMM_WORLD);
	// print_vec(vec_x, matr_size, comm_size, rank, "x");

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