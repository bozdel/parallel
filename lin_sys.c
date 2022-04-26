#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include "dbg.h"
// #include "misc.h"
#include "misc2.h"


#define FAST




void init_vecs_v1(double *vec_b, double *vec_x, int size, int shift) {
	for (int i = 0; i < size; i++) {
		vec_b[i] = i + 1 + shift;
		vec_x[i] = 0;
	}
}

// dots_num - dots where temperature != 0
void init_vecs_v3(double *vec_b, double *vec_x, int size, int dots_num) {
	for (int i = 0; i < size; i++) {
		vec_b[i] = 0;
		vec_x[i] = 0;
	}
	for (int i = 0; i < dots_num; i++) {
		int ind = rand() % size;
		vec_b[ind] = rand() % 100 - 50;
	}
}

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

/*
// returns cycles amont
// vec_x - destination vector
int solve_fast(double *part, int part_size, double *vec_b, double *vec_x, int size, double precision, int *displs, int *recvcounts, int rank) {
	int cycle = 0;

	double norm2_b = scalar_mul(vec_b, vec_b, size);

	// something wrong with allocation ( -np 4 ./prog 3 7)
	double *Ax = (double*)malloc(size * sizeof(double));
	double *vec_y = (double*)malloc(size * sizeof(double));
	double *Ay = (double*)malloc(size * sizeof(double));


	matr_mul(part, part_size, vec_x, size, recvcounts, displs, Ax); // Ax
	sub(Ax, vec_b, size, vec_y); // y = Ax - b
	for (cycle = 0 ; !check(vec_y, norm2_b, size, precision); cycle++) { // check - (||Ax-b||/||b||)^2 < precision^2
		matr_mul(part, part_size, vec_y, size, recvcounts, displs, Ay); // Ay

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

	matr_mul(part, part_size, vec_x, size, recvcounts, displs, vec_y);

	sub(vec_y, vec_b, size, vec_y);

	for (cycle = 0 ; !check(vec_y, norm2_b, size, precision); cycle++) {
		matr_mul(part, part_size, vec_y, size, recvcounts, displs, Ay);

		double yAy = scalar_mul(vec_y, Ay, size);
		double AyAy = scalar_mul(Ay, Ay, size);
		double tou = yAy / AyAy;

		subk(vec_x, tou, vec_y, size, vec_x); // new x

		matr_mul(part, part_size, vec_x, size, recvcounts, displs, vec_y);

		sub(vec_y, vec_b, size, vec_y); // new y
	}

	return cycle;
}
*/

int solve_fast(double *part, int part_size, double *vec_b, double *vec_x, int size, double precision, int *shifts, MPI_Comm ring, int rank, int comm_size, int *neighbours) {
	int cycle = 0;

	double norm2_b = scalar_mul_distr(vec_b, vec_b, size);

	// something wrong with allocation ( -np 4 ./prog 3 7)
	double *Ax = (double*)malloc(part_size * sizeof(double));
	double *vec_y = (double*)malloc(part_size * sizeof(double));
	double *Ay = (double*)malloc(part_size * sizeof(double));

	print_distr_vec(vec_b, part_size, comm_size, rank);
	print_distr_vec(vec_x, part_size, comm_size, rank);

	matr_mul(part, vec_x, Ax, part_size, size, shifts[rank], ring, comm_size, neighbours); // Ax
	print_distr_vec(Ax, part_size, comm_size, rank);
	sub(Ax, vec_b, part_size, vec_y); // y = Ax - b
	print_distr_vec(vec_y, part_size, comm_size, rank);
	for (cycle = 0 ; !check_distr(vec_y, norm2_b, part_size, precision); cycle++) { // check - (||Ax-b||/||b||)^2 < precision^2
		matr_mul(part, vec_y, Ay, part_size, size, shifts[rank], ring, comm_size, neighbours); // Ay
		printf("aloha\n");
		double yAy = scalar_mul_distr(vec_y, Ay, size);
		double AyAy = scalar_mul_distr(Ay, Ay, size);
		double tou = yAy / AyAy;
		printf("tou: %f\n", tou);

		sleep(1);

		subk(vec_x, tou, vec_y, part_size, vec_x); // new x = x - ty

		subk(Ax, tou, Ay, part_size, Ax); // new Ax. A(x - ty) = Ax - tAy
		sub(Ax, vec_b, part_size, vec_y); // new y. y = Ax - b
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

	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	// ----------creating ring communicator------------
	MPI_Comm ring_comm;
	int periods[1] = {true};
	int dims[1] = {comm_size};

	MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, false, &ring_comm);

	enum derictions {LEFT, RIGHT};
	int neighbours_ranks[2] = {0, 0};
	MPI_Cart_shift(ring_comm, 0, 1, &neighbours_ranks[LEFT], &neighbours_ranks[RIGHT]);

	for (int i = 0; i < comm_size; i++) {
		if (rank == i) {
			printf("(left, i, right): (%d, %d, %d)\n", neighbours_ranks[LEFT], rank, neighbours_ranks[RIGHT]);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	int part_size = part_size_by_rank(matr_size, comm_size, rank);
	int shift = shift_by_rank(matr_size, comm_size, rank);
	

	int *shifts = (int*)malloc(comm_size * sizeof(int));
	int *sizes = (int*)malloc(comm_size * sizeof(int));
	for (int proc_rank = 0; proc_rank < comm_size; proc_rank++) {
		shifts[proc_rank] = shift_by_rank(matr_size, comm_size, proc_rank);
		sizes[proc_rank] = part_size_by_rank(matr_size, comm_size, proc_rank);
	}

	double *part = (double*)malloc(part_size * matr_size * sizeof(double));
	init_matrix_v1(part, part_size, matr_size, shift);
	double *vec_b = (double*)malloc(part_size * sizeof(double));
	double *vec_x = (double*)malloc(part_size * sizeof(double));
	init_vecs_v1(vec_b, vec_x, part_size, shift);

	MPI_Barrier(MPI_COMM_WORLD);
	print_distr_vec(vec_b, part_size, comm_size, rank);
	MPI_Barrier(MPI_COMM_WORLD);

	/*double *recv_buff = (double*)malloc(sizes[0] * sizeof(double));
	double *send_buff = (double*)malloc(sizes[0] * sizeof(double));
	for (int i = 0; i < part_size; i++) send_buff[i] = vec_b[i];
	for (int i = 0; i < comm_size; i++) {
		int recv_size = rotate(send_buff, recv_buff, part_size, ring_comm, neighbours_ranks);
		printf("\n");
		MPI_Barrier(MPI_COMM_WORLD);
		print_distr_vec(recv_buff, recv_size, comm_size, rank);
		MPI_Barrier(MPI_COMM_WORLD);
		for (int i = 0; i < sizes[0]; i++) send_buff[i] = recv_buff[i];
	}

	// print_str("testing mul\n", MPI_COMM_WORLD, rank);
	printf("testing prog\n");
	print_matr(part, matr_size, part_size, comm_size, rank);

	double *dst = (double*)malloc(part_size * sizeof(double));
	// matr_mul(double *part_matr, double *part_vec, double *dst, int m_size, int v_size, int full_size, int init_size, MPI_Comm ring, int proc_amo, int *ring_neighbours);
	matr_mul(part, vec_b, dst, part_size, matr_size, shifts[rank], ring_comm, comm_size, neighbours_ranks);

	print_distr_vec(dst, part_size, comm_size, rank);

	double smul = scalar_mul_distr(vec_b, dst, part_size);
	printf("smul: %.3f\n", smul);*/

	int cycle = solve_fast(part, part_size, vec_b, vec_x, matr_size, precision, shifts, ring_comm, rank, comm_size, neighbours_ranks);
	if (rank == 0) printf("cycles: %d\n", cycle);

	print_distr_vec(vec_x, part_size, comm_size, rank);

/*
	// --------initing vecs----------

	double *vec_b = (double*)malloc(matr_size * sizeof(double));
	double *vec_x = (double*)malloc(matr_size * sizeof(double));

	int dots_num = 6;
	init_vecs_v3(vec_b, vec_x, matr_size, dots_num);


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
	// print_vec(vec_x, matr_size, comm_size, rank, "x");*/

	MPI_Finalize();
	return 0;
}