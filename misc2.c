void sub_matr_mul(double *part_matr, double *part_vec, double *dst, int m_size, int v_size, int whence) {
	for (int i = 0; i < m_size; i++) {
		double *line = part_matr + i * size + whence;
		double tmp = 0.0;
		for (int j = 0; j < v_size; j++) {
			tmp += line[j] * part_vec[j];
		}
		dst[i] = tmp;
	}
}

// returns recieved buffer size
int rotate(double *send, double *recv, int send_size) {
	int recv_size = 0;
	MPI_Sendrecv(send, send_size, MPI_DOUBLE, neighbours[LEFT], 0,
				 recv, send_size + 1, MPI_DOUBLE, neighbours[RIGHT], MPI_ANY_TAG, ring_comm, MPI_STATUS_IGNORE);
	
}

void matr_mul(double *part_matr, double *part_vec, double *dst, int m_size, int v_size) {
	int recieved_size = v_size;
	int accum_size = init_size;
	for (int i = 0; i < proc_amo; i++) {
		sub_matr_mul(part_matr, part_vec, dst, m_size, v_size, accum_size);
		
		recieved_size = rotate(part_vec, tmp, recieved_size);
		accum_size = (accum_size + recieved_size) % full_size;
	}
}