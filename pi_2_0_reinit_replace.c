/* The 1st version of the sample of REINIT++ fault tolerance platform
 * Processes, that were killed, will be replaced and their share of work
 * will be compensated and distributed among survived processes and the new replacements .
 *
 * to run: mpirun -np 4 --mca orte_enable_recovery 1 ./out.o 10000 0
     * -np number of processes
     * --mca orte_enable_recovery  activates the fault tolerance in REINIT++
     * 10000  amount of points
     * 0 is the kill mode   1:active or 0:inactive
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <unistd.h>
#include <signal.h>

#define RANDOMIZE_XY(x, y) *x=(-1.0 + (rand() / (RAND_MAX / 2.0))); *y=(-1.0 + (rand() / (RAND_MAX / 2.0)));
#define INSIDE_OF_CIRCLE(x, y) (x * x) + (y * y)<= 1

#define IS_ROOT (me == 0)

#define IS_NEW (state_gl == OMPI_REINIT_NEW)
#define IS_RESTARTED (state_gl == OMPI_REINIT_RESTARTED)
#define IS_REINITED (state_gl == OMPI_REINIT_REINITED)

/*############### help functions ###############*/

void kill_me(int kill_counter);

void check_args(char *argv[]);

/*--------------------------------------------*/


/*############### FT-Platform variables ###############*/
//amount of starting processes
int total;

//id/rank of current prozess
int me;

double time;
/*--------------------------------------------*/


/*############### monte pi calculation variables ###############*/
//total calculated amount of produced points, which happens to be inside the circle
double circle_pts;

//partial calculated amount of points within the circle in this process
double circle_pts_local;

//total amount of points to be produced
double all_pts_init, all_pts;

//the estimated pi
double pi;

//point's coordinates
double x, y;
/*--------------------------------------------*/


/*############### help variables ###############*/
double done_pts, done_local;
int kill_mode, already_killed;

//state of a process (3 enums)
// OMPI_REINIT_NEW = 1, OMPI_REINIT_REINITED, OMPI_REINIT_RESTARTED,
// needed to access the process state from inside help functions
OMPI_reinit_state_t state_gl;

//to determine if there is a need to sync values to the failed processes
int failure = 0, local_failure = 0;

int wrong_input_from = 0;

/*--------------------------------------------*/


int resilient_main(int argc, char *argv[], OMPI_reinit_state_t state) {

    MPI_Comm current_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &current_comm);
    MPI_Comm_rank(current_comm, &me);
    MPI_Comm_size(current_comm, &total);
    state_gl = state;
    check_args(argv);
    if (!wrong_input_from) {
        all_pts_init = atof(argv[1]);
        if (IS_NEW) {
            all_pts = all_pts_init;
            time = MPI_Wtime();
            srand(me);
        }

        kill_mode = atoi(argv[2]);

        do {
            if (IS_RESTARTED && !already_killed) {
                local_failure = 1;
                already_killed = 1;
                srand(me * 10);

            } else {
                local_failure = 0;
            }
            //kill_me(1);
            MPI_Allreduce(&local_failure, &failure, 1, MPI_INT, MPI_SUM, current_comm);
            if (!failure) {
                done_local = 0;
                for (int i = 0; i < (int) (all_pts / total) + ((IS_ROOT) ? ((int) all_pts % total) : 0); ++i) {
                    //generate randoms in range [-1.0 to 1.0] both inclusive
                    RANDOMIZE_XY(&x, &y)
                    if (INSIDE_OF_CIRCLE(x, y)) {
                        ++circle_pts_local;
                    if (i > (int) all_pts / (total * 2)) kill_me(1);
                    }
                    done_local++;
                }
            }

            MPI_Reduce(&done_local, &done_pts, 1, MPI_DOUBLE, MPI_SUM, 0,current_comm);
            all_pts -= done_pts;
            MPI_Bcast(&all_pts, 1, MPI_DOUBLE, 0, current_comm);
            //kill_me(2);
        } while (all_pts > 0 || failure);

        MPI_Reduce(&circle_pts_local, &circle_pts, 1, MPI_DOUBLE, MPI_SUM, 0, current_comm);

        if (IS_ROOT) {
            MPI_Comm_size(current_comm, &total);

            pi = 4.0 * circle_pts / all_pts_init;
            printf("\npts: %.lf\npi: %lf\ntime: %.1fs\nstart procs: %d\n",
                   all_pts_init, pi, MPI_Wtime() - time, total);
        }
        MPI_Barrier(current_comm);
    }
    return 0;

}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    OMPI_Reinit(argc, argv, resilient_main);
    MPI_Finalize();
    return 0;

}


/**
 * guarantees that all required args are met and cancels with a message otherwise
 * @param argv  passed from main args from the user
 */
void check_args(char *argv[]) {

    if (!argv[1] || !argv[2]) {
        wrong_input_from = 1;
        if (IS_ROOT) {
            printf("Specify total number of points and the kill mode(0 or else)\ne.g.: $ mpirun -np %d --mca orte_enable_recovery 1 %s 10000000 0\n",
                   total, argv[0]);
        }
    }
}

/**
 * kills a process counting from the last process id downwards if kill_mode is active and it was not the root process
 * @param kill_counter starting top-down 1 (total-1), 2 (total-2), 3 (total-3), ...
 */
void kill_me(int kill_counter) {

    if (kill_mode && !already_killed && !IS_ROOT && me == total - kill_counter) {
        printf("\033[0;31mkilled: p(%d)\033[0m, ", me);
        fflush(stdout);
        raise(SIGKILL);
    }
}