#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include "misc2.h"

#include "dbg.h"


int solve_fast(double *part, int part_size, double *vec_b, double *vec_x, int size, double precision, int *shifts, MPI_Comm ring, int rank, int comm_size, int *neighbours) {
	int cycle = 0;

	double norm2_b = scalar_mul_distr(vec_b, vec_b, part_size);

	double *Ax = (double*)malloc(part_size * sizeof(double));
	double *vec_y = (double*)malloc(part_size * sizeof(double));
	double *Ay = (double*)malloc(part_size * sizeof(double));

	matr_mul(part, vec_x, Ax, part_size, size, shifts[rank], ring, comm_size, neighbours); // Ax
	sub(Ax, vec_b, part_size, vec_y); // y = Ax - b
	for (cycle = 0 ; !check_distr(vec_y, norm2_b, part_size, precision); cycle++) { // check - (||Ax-b||/||b||)^2 < precision^2
		matr_mul(part, vec_y, Ay, part_size, size, shifts[rank], ring, comm_size, neighbours); // Ay

		double yAy = scalar_mul_distr(vec_y, Ay, part_size);
		double AyAy = scalar_mul_distr(Ay, Ay, part_size);
		double tou = yAy / AyAy;

		subk(vec_x, tou, vec_y, part_size, vec_x); // new x = x - ty
		subk(Ax, tou, Ay, part_size, Ax); // new Ax. A(x - ty) = Ax - tAy
		sub(Ax, vec_b, part_size, vec_y); // new y. y = Ax - b
	}

	free(Ax);
	free(vec_y);
	free(Ay);

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

	// ----------creating ring communicator------------
	MPI_Comm ring_comm;
	int periods[1] = {true};
	int dims[1] = {comm_size};

	MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, false, &ring_comm);

	int neighbours_ranks[2] = {0, 0};
	MPI_Cart_shift(ring_comm, 0, 1, &neighbours_ranks[LEFT], &neighbours_ranks[RIGHT]);

	
	// -----------initing some information-----------------
	int part_size = part_size_by_rank(matr_size, comm_size, rank);
	int shift = shift_by_rank(matr_size, comm_size, rank);
	

	int *shifts = (int*)malloc(comm_size * sizeof(int));
	for (int proc_rank = 0; proc_rank < comm_size; proc_rank++) {
		shifts[proc_rank] = shift_by_rank(matr_size, comm_size, proc_rank);
	}

	// --------------iniing data--------------------
	double *part = (double*)malloc(part_size * matr_size * sizeof(double));
	double *vec_b = (double*)malloc(part_size * sizeof(double));
	double *vec_x = (double*)malloc(part_size * sizeof(double));

	// init_matrix_v3(part, part_size, matr_size, shift, Nx, Ny);
	init_matrix_v1(part, part_size, matr_size, shift);

	init_vecs_v3_distr(vec_b, vec_x, matr_size, part_size, dots_num, shift);

	//---------------main part-------------------
	clock_t beg = clock();
	int cycle = solve_fast(part, part_size, vec_b, vec_x, matr_size, precision, shifts, ring_comm, rank, comm_size, neighbours_ranks);
	clock_t end = clock();

	if (rank == 0) printf("cycles: %d\n", cycle);
	printf("rank %d, time: %f\n", rank, (float)(end - beg) / CLOCKS_PER_SEC);

	print_distr_vec(vec_x, part_size, comm_size, rank, "x");


	MPI_Finalize();
	return 0;
}
