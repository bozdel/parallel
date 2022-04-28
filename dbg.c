#include "dbg.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void print_str(char const* string, int comm_size, int rank) {
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < comm_size; i++) {
		if (rank == i) {
			printf("%d %s\n", rank, string);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
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

void print_vec(double *vec, int vec_size, int comm_size, int rank, char const* string) {
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < comm_size; i++) {
		if (rank == i) {
			if (string) {
				printf("%s, rank: %d:\n", string, rank);
			}
			else {
				printf("%d: ", rank);
			}
			for (int i = 0; i < vec_size; i++) {
				printf("%.3f ", vec[i]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void print_distr_vec(double *vec, int size, int comm_size, int rank, char const* string) {
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) printf("%s: ", string);
	for (int i = 0; i < comm_size; i++) {
		if (rank == i) {
			for (int j = 0; j < size; j++) {
				printf("%.3f ", vec[j]);
			}
			// printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	printf("\n");
}

void print_part(double *part, int matr_size, int part_size) {
	for (int i = 0; i < part_size; i++) {
		for (int j = 0; j < matr_size; j++) {
			printf("%.0f ", part[i * matr_size + j]);
		}
		printf("\n");
	}
	// printf("\n");
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

void gen_randvec(double *dst, int size) {
	for (int i = 0; i < size; i++) {
		dst[i] = (double)(rand() % 100);
	}
}

void print_vec_0(double *vec, int size, int rank, char const* string) {
	if (rank == 0) {
		if (string) {
			printf("%s: ", string);
		}
		for (int i = 0; i < size; i++) {
			printf("%.3f ", vec[i]);
		}
		printf("\n");
	}
}