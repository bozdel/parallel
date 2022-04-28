#include "misc2.h"
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "dbg.h"

void sub_matr_mul(double *part_matr, double *part_vec, double *dst, int m_size, int v_size, int full_size, int whence, int rank) {
	for (int i = 0; i < m_size; i++) {
		double *line = part_matr + i * full_size + whence;
		double tmp = 0.0;
		for (int j = 0; j < v_size; j++) {
			// if (rank == 0) printf("%f, ", line[j]);
			tmp += line[j] * part_vec[j];
		}
		// if (rank == 0) printf("\n");
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