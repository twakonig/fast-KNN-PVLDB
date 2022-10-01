#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <papi.h>

#ifndef TEAM36_PAPI_WRAPPER_H
#define TEAM36_PAPI_WRAPPER_H

void init_lib() {
    int retval;
    retval=PAPI_library_init(PAPI_VER_CURRENT);
    if (retval!=PAPI_VER_CURRENT) {
        fprintf(stderr,"Error initializing PAPI! %s\n",
                PAPI_strerror(retval));
    }
}

void create_eventset(int *eventset) {
    int retval;
    retval=PAPI_create_eventset(eventset);
    if (retval!=PAPI_OK) {
        fprintf(stderr,"Error creating eventset! %s\n",
                PAPI_strerror(retval));
    }
}

// alternative events: PAPI_TOT_CYC, PAPI_DP_OPS
void add_flopcount(int eventset) {
    int retval;
    retval=PAPI_add_named_event(eventset, "PAPI_DP_OPS");
    if (retval!=PAPI_OK) {
        fprintf(stderr,"Error adding PAPI_DP_OPS: %s\n",
                PAPI_strerror(retval));
    }
}

// reset counter and start eventset
void start_ctr(int *eventset) {
    int retval;
    PAPI_reset(*eventset);
    retval=PAPI_start(*eventset);
    if (retval!=PAPI_OK) {
        fprintf(stderr,"Error starting CUDA: %s\n",
                PAPI_strerror(retval));
    }
}

// stop and print counter value
void stop_ctr(int *eventset, long long *counter, char* funcname) {
    int retval;
    retval=PAPI_stop(*eventset,counter);
    if (retval!=PAPI_OK) {
        fprintf(stderr,"Error stopping:  %s\n",
                PAPI_strerror(retval));
    }
    else {
        printf("Measured %lld flops in %s\n",*counter, funcname);
    }
}

#endif //TEAM36_PAPI_WRAPPER_H
