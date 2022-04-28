#include "misc2.h"
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "dbg.h"

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

void init_vecs_v3_distr(double *vec_b, double *vec_x, int full_size, int part_size, int dots_num, int shift) {
	double *tmp_b = (double*)malloc(full_size * sizeof(double));
	double *tmp_x = (double*)malloc(full_size * sizeof(double));
	init_vecs_v3(tmp_b, tmp_x, full_size, dots_num);
	for (int i = 0; i < part_size; i++) {
		vec_b[i] = tmp_b[i + shift];
		vec_x[i] = tmp_x[i + shift];
	}
	free(tmp_b);
	free(tmp_x);
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

void sub_matr_mul(double *part_matr, double *part_vec, double *dst, int m_size, int v_size, int full_size, int whence, int rank) {
	for (int i = 0; i < m_size; i++) {
		double *line = part_matr + i * full_size + whence;
		double tmp = 0.0;
		for (int j = 0; j < v_size; j++) {
			tmp += line[j] * part_vec[j];
		}
		dst[i] += tmp;
	}
}

// clockwise rotating
// returns received buffer size
int rotate(double *send, double *recv, int send_size, MPI_Comm ring, int *ring_neighbours) {
	int recv_size = 0;
	MPI_Status status;
	MPI_Sendrecv(send, send_size, MPI_DOUBLE, ring_neighbours[LEFT], 0,
				 recv, send_size + 1, MPI_DOUBLE, ring_neighbours[RIGHT], MPI_ANY_TAG, ring, &status);
	MPI_Get_count(&status, MPI_DOUBLE, &recv_size);
	return recv_size;
}


// proc_amo - ring_comm size
void matr_mul(double *part_matr, double *part_vec, double *dst, int pv_size, int full_size, int init_shift, MPI_Comm ring, int proc_amo, int *ring_neighbours) {
	int accum_shift = init_shift;
	for (int i = 0; i < pv_size; i++) {
		dst[i] = 0.0;
	}

	// initing data for first cycle
	int received_size = pv_size;
	// fix pv_size + 1. it's a bit different for procs (but still should work)
	double *recv_buff = (double*)malloc((pv_size + 1) * sizeof(double));
	double *send_buff = (double*)malloc((pv_size + 1) * sizeof(double));
	memcpy(send_buff, part_vec, received_size * sizeof(double));

	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	for (int i = 0; i < proc_amo; i++) {
		sub_matr_mul(part_matr, send_buff, dst, pv_size, received_size, full_size, accum_shift, rank);
		
		accum_shift = (accum_shift + received_size) % full_size;
		received_size = rotate(send_buff, recv_buff, received_size, ring, ring_neighbours);
		
		memcpy(send_buff, recv_buff, received_size * sizeof(double));
	}
}

double scalar_mul_distr(double *vec1, double *vec2, int size) {
	double sub_res = 0;
	for (int i = 0; i < size; i++) {
		sub_res += vec1[i] * vec2[i];
	}
	double res = 0;
	MPI_Allreduce(&sub_res, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
bool check_distr(double *vec, double norm2_b, int size, double precision) {
	double norm2_v = scalar_mul_distr(vec, vec, size);
	return norm2_v / norm2_b < (precision * precision);
}

void sub(double *vec1, double *vec2, int size, double *dst_vec) {
	for (int i = 0; i < size; i++) {
		dst_vec[i] = vec1[i] - vec2[i];
	}
}

void subk(double *vec1, double k, double *vec2, int size, double *dst) {
	for (int i = 0; i < size; i++) {
		dst[i] = vec1[i] - k * vec2[i];
	}
}