/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 29, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -fopenmp -std=c99 -Wall -O3 -lm 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s matrix-size num_threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
		fprintf(stderr, "num_threads: Number of parallel threads\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);
	int num_threads = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */

	struct timeval start, stop;
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));
	
	/* Compute the Jacobi solution using openMP. 
     * Solution is returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using omp\n");
	gettimeofday(&start, NULL);
	compute_using_omp(A, mt_solution_x, B, num_threads);
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
	gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using openMP. 
 * Result must be placed in mt_sol_x. */
void compute_using_omp(const matrix_t A, matrix_t mt_sol_x, const matrix_t B, int num_threads)
{
	int max_iter = 100000;
    int i, j;
    int num_rows = A.num_rows;
    int num_cols = A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      
    
    /* Initialize current jacobi solution. */
    for (i = 0; i < num_rows; i++)
        mt_sol_x.elements[i] = B.elements[i];

    /* Perform Jacobi iteration. */
    int done = 0;
    double ssd, mse, sum;
    int num_iter = 0;
	double* partial_ssd = (double *)malloc(num_threads * sizeof(double));
    int tid;
    
	#pragma omp parallel num_threads(num_threads) private(i,j,sum, ssd, tid) default(shared)
	{ /* -- START of parallel region -- */

	tid  = omp_get_thread_num();
    while (!done) {
		partial_ssd[tid] = 0.0;
		#pragma omp for private(i,j,sum)
        for (i = 0; i < num_rows; i++) {
            sum = 0.0;
            for (j = 0; j < num_cols; j++) {
                if (i != j){
                    sum += A.elements[i * num_cols + j] * mt_sol_x.elements[j];
				}
            }
           
            /* Update values for the unkowns for the current row. */
            new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
			partial_ssd[tid] += (new_x.elements[i] - mt_sol_x.elements[i]) * (new_x.elements[i] - mt_sol_x.elements[i]);
        }

        /* Check for convergence and update the unknowns. */
		#pragma omp for private(i)
        for (i = 0; i < num_rows; i++) {
            mt_sol_x.elements[i] = new_x.elements[i];
        }


		#pragma omp single
		{
        ssd = 0.0; 
		for (i=0; i< num_threads; i++){
			ssd += partial_ssd[i];
		}
        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        // fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
        
        if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;
		}
    }
	}  /* -- END of parallel region -- */

	if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");

	free(new_x.elements);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



