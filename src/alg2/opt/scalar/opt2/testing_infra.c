#include "opt2.h"
#include "read_input.h"
#include "mat.h"

#define PATH "../../../../../data/"
#define FEATURE_TRAIN "features_training.npy"
#define LABEL_TRAIN "labels_training.npy"
#define FEATURE_TEST "features_testing.npy"
#define LABEL_TEST "labels_testing.npy"
#define PATH_FEATURE_TRAIN PATH FEATURE_TRAIN
#define PATH_FEATURE_TEST PATH FEATURE_TEST
#define PATH_LABEL_TRAIN PATH LABEL_TRAIN
#define PATH_LABEL_TEST PATH LABEL_TEST


int main(int argc, char** argv) {
    if (argc < 5){

      printf("No/not enough arguments given, please input N M d K");
      return 0;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int d = atoi(argv[3]);
    int K = atoi(argv[4]);

    int use_random_data = 0;
    if (argc == 6){
        use_random_data = atoi(argv[5]);
    }

    mat sp_approx;
    mat x_trn;
    mat x_tst;
    int* y_trn = malloc(N*sizeof(float));
    int* y_tst = malloc(M*sizeof(float));

    build(&x_trn, N, d);
    build(&x_tst, M, d);
    build(&sp_approx, M, N);

    if(use_random_data){
        initialize_rand_mat(&x_trn);
        initialize_rand_mat(&x_tst);
        initialize_rand_array(y_trn, N);
        initialize_rand_array(y_tst, M);
    }
    else{
        read_input_features(x_trn.data, PATH_FEATURE_TRAIN, N, d);
        read_input_labels(y_trn, PATH_LABEL_TRAIN, N, 1);
        read_input_features(x_tst.data, PATH_FEATURE_TEST, M, d);
        read_input_labels(y_tst, PATH_LABEL_TEST, M, 1);
    }

    for(int i = 0; i < sp_approx.n1; i++){
        for(int j = 0; j < sp_approx.n2; j++){
            mat_set(&sp_approx, i, j, 0.0);
        }
    }

    srand(42); // fix seed for RNG
    knn_mc_approximation(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, K, 128);

    // for(int i = 0; i < sp_approx.n2; i++){
    //   for(int j = 0; j < sp_approx.n1; j++){
    //     printf("%.8f ", mat_get(&sp_approx, i, j));
    //   }
    //   printf("\n");
    // }

    // printf("\n");
    // printf("\n");
    // printf("\n");
    float sp[N];
    for(int i = 0; i < sp_approx.n2; i++){
      sp[i] = 0;
      for(int j = 0; j < sp_approx.n1; j++){
        sp[i] += mat_get(&sp_approx, j, i);
      }
      sp[i] /= M;
      if(i < sp_approx.n2-1){
        printf("%.8f ", sp[i]);
      }
      else{
        printf("%.8f", sp[i]);
      }
    }
    destroy(&sp_approx);
    free(y_trn);
    free(y_tst);
    destroy(&x_trn);
    destroy(&x_tst);

    return 0;
}

