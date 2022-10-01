#include "alg1.h"

//#define N 512

int main(int argc, char** argv) {
    if (argc != 2){

        printf("No/not enough arguments given, please input N\n");
        return -1;
    }
    int N = atoi(argv[1]);

    int_mat x_tst_knn_gt;
    mat sp_gt;
    mat x_trn;
    mat x_tst;
    int* y_trn = malloc(N*sizeof(int));
    int* y_tst = malloc(N*sizeof(int));

    build(&x_trn, N, N);
    build(&x_tst, N, N);
    build(&sp_gt, N, N);
    build_int_mat(&x_tst_knn_gt, N, N);

    // randomly initialize all data containers
    initialize_rand_mat(&x_trn);
    initialize_rand_mat(&x_tst);
    initialize_mat(&sp_gt, 0.0);
    initialize_rand_array(y_trn, N);
    initialize_rand_array(y_tst, N);

    get_true_knn(&x_tst_knn_gt, &x_trn, &x_tst);
    compute_single_unweighted_knn_class_shapley(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, 1);

    destroy(&sp_gt);
    destroy(&x_trn);
    destroy(&x_tst);
    destroy_int_mat(&x_tst_knn_gt);
    free(y_trn);
    free(y_tst);
    return 0;
}
