#ifndef MISC
#define MISC

#include <stdbool.h>

void part_matr_mul(double *part_matr, int part_size, double *vec, int size, double *dest, int shift);

// size - size of matrix and vector (they should be same)
void matr_mul(double *part_matr, int part_size, double *vec, int size, int shift, int *recvcounts, int *displs, double *dst_vec/*, int rank, int comm_size*/);

void sub(double *vec1, double *vec2, int size, double *dst_vec);

void kvec(double *vec, double k, int size, double *dst);

void subk(double *vec1, double k, double *vec2, int size, double *dst);

// something near scalmul_allred(vec1, vec2 + my_shift, part_size)
double scalar_mul_allred(double *vec1, double *vec2, int size);

double scalar_mul(double *vec1, double *vec2, int size);

int part_size_by_rank(int matr_size, int comm_size, int rank);

// index of the first line of part in full matrix
int shift_by_rank(int matr_size, int comm_size, int rank);

bool check(double *vec, double norm2_b, int size, double precision);

#endif