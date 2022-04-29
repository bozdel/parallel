#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include "dbg.h"
#include "misc.h"


#define FAST



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
	int dots_num = 6; // dots where temperature != 0
	if (argc >= 5) {
		dots_num = atoi(argv[4]);
	}

	int err_code;

	if ((err_code = MPI_Init(&argc, &argv)) != 0) {
		printf("error\n");
		return err_code;
	}


	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	
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
	double *vec_b = (double*)malloc(matr_size * sizeof(double));
	double *vec_x = (double*)malloc(matr_size * sizeof(double));
	
	// init_vecs_v1(vec_b, vec_x, matr_size, matr_size);
	init_vecs_v3(vec_b, vec_x, matr_size, dots_num);

	// init_matrix_v1(part, part_size, matr_size, my_shift);
	init_matrix_v3(part, part_size, matr_size, my_shift, Nx, Ny);



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

	print_vec_0(vec_x, matr_size, rank, "x");


	MPI_Finalize();
	return 0;
}