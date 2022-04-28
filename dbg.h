#ifndef DBG
#define DBG

void print_str(char const* string, int comm_size, int rank);

void print_vecint(int *vec, int vec_size, int comm_size, int rank);

void print_vec(double *vec, int vec_size, int comm_size, int rank, char const* string);

void print_distr_vec(double *vec, int size, int comm_size, int rank, char const* string);

void print_part(double *part, int matr_size, int part_size);

void print_matr(double *part, int matr_size, int part_size, int comm_size, int rank);

void gen_randvec(double *dst, int size);

void print_vec_0(double *vec, int size, int rank, char const* string);

#endif