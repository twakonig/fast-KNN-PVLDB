#include "alg2.h"

int main() {

    mat sp_approx;
    mat x_trn;
    mat x_tst;

    int* y_trn = malloc(4*sizeof(float));
    int* y_tst = malloc(2*sizeof(float));

    y_trn[0] = 1;
    y_trn[1] = 0;
    y_trn[2] = 1;
    y_trn[3] = 0;

    y_tst[0] = 0;
    y_tst[1] = 1;

    build(&x_trn, 4, 5);
    build(&x_tst, 2, 5);
    build(&sp_approx, 2, 4);

    float ctr = 0.0;
    // initialize some input
    for(int i = 0; i < x_trn.n1; i++){
        for(int j = 0; j < x_trn.n2; j++){
            mat_set(&x_trn, i, j, ctr);
            ctr++;
        }
    }

    ctr = 0.0;
    for(int i = 0; i < x_tst.n1; i++){
        for(int j = 0; j < x_tst.n2; j++){
            mat_set(&x_tst, i, j, ctr);
            ctr++;
        }
    }

    for(int i = 0; i < sp_approx.n1; i++){
        for(int j = 0; j < sp_approx.n2; j++){
            mat_set(&sp_approx, i, j, 0.0);
        }
    }

    srand((unsigned) time(NULL));
    knn_mc_approximation(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, 1, 130);

    printf("sp_approx\n");
    for(int i = 0; i < sp_approx.n1; i++){
      for(int j = 0; j < sp_approx.n2; j++){
        printf(" %.15f", mat_get(&sp_approx, i, j));
      }
      printf("\n");
    }
    destroy(&sp_approx);
    destroy(&x_trn);
    destroy(&x_tst);
    free(y_trn);
    free(y_tst);    
    return 0;
}
