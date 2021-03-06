#include "misc.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdbool.h>

void init_vecs_v1(double *vec_b, double *vec_x, int size, int full_size) {
	for (int i = 0; i < size; i++) {
		vec_b[i] = full_size + 1;
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
	double **matr = (double**)malloc(part_size * sizeof(double*));
	for (int i = 0; i < part_size; i++) {
		matr[i] = part + i * matr_size;
	}
	for (int i = 0; i < part_size; i++) {
		matr[i][i + my_shift] = 4; // row (-4)
		if ( (i + my_shift + 1 < matr_size) && ((i + 1 + my_shift) % Nx != 0) ) {
			matr[i][i + my_shift + 1] = 1; // upper shifted (1) row
		}
		if ( (i + my_shift - 1 >= 0) && ((i + my_shift) % Nx != 0) ) {
			matr[i][i + my_shift - 1] = 1; // lower shifted (1) row
		}
		if (i + my_shift + Nx < matr_size) {
			matr[i][i + my_shift + Nx] = 1; // upper2 (1) row
		}
		if (i + my_shift - Nx >= 0) {
			matr[i][i + my_shift - Nx] = 1; // lower2 (1) row
		}
	}
	free(matr);
}

void part_matr_mul(double *part_matr, int part_size, double *vec, int size, double *dest) {
	for (int i = 0; i < part_size; i++) {
		double *line = part_matr + i * size;
		double tmp_res = 0.0;
		for (int j = 0; j < size; j++) {
			tmp_res += line[j] * vec[j];
		}
		dest[i] = tmp_res;
	}
}

// size - size of matrix and vector (they should be same)
void matr_mul(double *part_matr, int part_size, double *vec, int size, int *recvcounts, int *displs, double *dst_vec, double *tmp/*, int rank, int comm_size*/) {
	part_matr_mul(part_matr, part_size, vec, size, tmp);
	// print_vec(tmp, part_size, comm_size, rank);


	MPI_Allgatherv(tmp, part_size, MPI_DOUBLE, dst_vec, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
}

void sub(double *vec1, double *vec2, int size, double *dst_vec) {
	for (int i = 0; i < size; i++) {
		dst_vec[i] = vec1[i] - vec2[i];
	}
}

void kvec(double *vec, double k, int size, double *dst) {
	for (int i = 0; i < size; i++) {
		dst[i] = k * vec[i];
	}
}

void subk(double *vec1, double k, double *vec2, int size, double *dst) {
	for (int i = 0; i < size; i++) {
		dst[i] = vec1[i] - k * vec2[i];
	}
}

// something near scalmul_allred(vec1, vec2 + my_shift, part_size)
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
	double res = 0.0;
	for (int i = 0; i < size; i++) {
		res += vec1[i] * vec2[i];
	}
	return res;
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

// squaring is correct because norm always > 0 and precision < 0 is meanless (so it > 0 too)
bool check(double *vec, double norm2_b, int size, double precision) {
	double norm2_v = scalar_mul(vec, vec, size);
	return norm2_v / norm2_b < (precision * precision);
}