#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include <config.h>
#include <dispatcher.h>
#include <opencl.h>
#include <calc.h>

#define SQR(x) ((x) * (x))

const char *load_source(const char *fname)
{
    FILE *f = fopen(fname, "rb");
    if (f == NULL)
    {
        printf("Can not open file [%s]. Exiting.\n", fname);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *source = malloc(fsize + 1);
    fread(source, 1, fsize, f);
    fclose(f);
    source[fsize] = 0;

    return source;
}

int file_lines(FILE *f)
{
    int num_lines = 0;
    fseek(f, 0, SEEK_SET);
    while (!feof(f))
    {
        char c = fgetc(f);
        if (c == '\n')
            num_lines++;
    }
    return num_lines;
}

size_t load_rays(const char *input_fname, real **pos, real **dir, cl_int **finished)
{
    int i;
    char line[1024];
    FILE *input = fopen(input_fname, "rt");
    int num_objects = file_lines(input) - 1;

    *pos = malloc(sizeof(real) * DIM * num_objects);
    *dir = malloc(sizeof(real) * DIM * num_objects);
    *finished = malloc(sizeof(cl_int) * num_objects);

    fseek(input, 0, SEEK_SET);
    fgets(line, 1024, input);
    for (i = 0; i < num_objects; i++)
    {
        fgets(line, 1024, input);
        double cpos[DIM];
        double cdir[DIM];
        sscanf(line, "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf", &cpos[0], &cpos[1], &cpos[2], &cpos[3], &cdir[0], &cdir[1], &cdir[2], &cdir[3]);
        int j;
        for (j = 0; j < DIM; j++)
        {
            (*pos)[i * DIM + j] = cpos[j];
            (*dir)[i * DIM + j] = cdir[j];
        }
    }
    fclose(input);
    for (i = 0; i < num_objects; i++)
        (*finished)[i] = 0;
}

void worker_function(struct opencl_state_s *opencl_state,
                     struct dispatcher_s *dispatcher,
                     int platform_id,
                     int device_id,
                     real T, real h,
                     real *args,
                     size_t num_args,
                     int num_steps)
{
    while (dispatcher_has_data(dispatcher))
    {
        real *bpos, *bdir;
        cl_int *bfinished;
        FILE **boutput;

        int amount = opencl_state->units[platform_id][device_id].max_parallel_points;

        size_t num_objects_in_block = dispatcher_get_next_block(dispatcher, &bpos, &bdir, &bfinished, &boutput, amount);
        if (num_objects_in_block == 0)
        {
            break;
        }

        perform_calculation(&opencl_state->units[platform_id][device_id],
                            T, h, bpos, bdir, bfinished, num_objects_in_block,
                            args, num_args, boutput, num_steps);
    }
}

struct worker_s
{
    struct opencl_state_s *opencl_state;
    struct dispatcher_s *dispatcher;
    int platform_id;
    int device_id;
    real T;
    real h;
    real *args;
    size_t num_args;
    int num_steps;
};

void *worker_launcher(void *args)
{
    struct worker_s *worker = args;

    worker_function(worker->opencl_state,
                    worker->dispatcher,
                    worker->platform_id,
                    worker->device_id,
                    worker->T,
                    worker->h,
                    worker->args,
                    worker->num_args,
                    worker->num_steps);
    return NULL;
}

int main(int argc, const char **argv)
{
    int i;

    if (argc < 8)
    {
        printf("Usage: geodesic2 input.csv output.csv metric.cl args.csv <T> <h> <num steps>\n");
        return 1;
    }

    char line[1024];
    struct opencl_state_s opencl_state;

    // Simulation
    double T;
    double h;
    int num_steps;

    sscanf(argv[5], "%lf", &T);
    sscanf(argv[6], "%lf", &h);
    sscanf(argv[7], "%i", &num_steps);

    // Init viewer light
    // load it from file

    const char *input_fname = argv[1];
    const char *output_fname = argv[2];
    const char *metric_fname = argv[3];
    const char *args_fname = argv[4];

    const char *out_dirname = NULL;
    if (argc >= 9)
    {
        out_dirname = argv[8];
    }

    /* Read arguments */
    FILE *af = fopen(args_fname, "rt");
    int num_args = file_lines(af) - 1;

    real *args = malloc(sizeof(real) * num_args);

    fseek(af, 0, SEEK_SET);
    fgets(line, 1024, af);
    for (i = 0; i < num_args; i++)
    {
        double v;
        fgets(line, 1024, af);
        sscanf(line, "%lf", &v);
        args[i] = v;
    }
    fclose(af);

    /* Read initial state */
    real *pos, *dir;
    cl_int *finished;
    cl_int num_objects = load_rays(input_fname, &pos, &dir, &finished);
    printf("Loaded %i objects\n", num_objects);

    /* Open output files */
    FILE **output_rays = NULL;
    if (out_dirname != NULL)
    {
        output_rays = malloc(sizeof(FILE*)*num_objects);
        for (i = 0; i < num_objects; i++)
        {
            char fname[4096];
            snprintf(fname, 4096, "%s/%05i.csv", out_dirname, i);
            output_rays[i] = fopen(fname, "wt");
        }
    }

    /* Init opencl platforms */
    size_t global;
    size_t local;

    const char *source_fname = BINROOT "/geodesic.cl";
    const char *source = load_source(source_fname);
    const char *metric = load_source(metric_fname);
    char *kernel_source = malloc(strlen(source) + strlen(metric) + 3);
    strcpy(kernel_source, source);
    strcat(kernel_source, metric);

    init_opencl(&opencl_state);
    init_opencl_program(&opencl_state, kernel_source);

    int platform_id = -1;
    int device_id = 0;

    for (i = 0; i < opencl_state.num_platforms; i++)
    {
        if (opencl_state.num_devices[i] > 0)
        {
            platform_id = i;
            device_id = 0;
            break;
        }
    }

    if (platform_id == -1)
    {
        printf("No platform\n");
        exit(0);
    }

    printf("Select platform %i, device %i\n", platform_id, device_id);

    struct dispatcher_s dispatcher;
    dispatcher_init(&dispatcher, pos, dir, finished, output_rays, num_objects);

    struct worker_s workers[MAX_DEVICES * MAX_PLATFORMS];
    pthread_t threads[MAX_DEVICES * MAX_PLATFORMS];
    size_t num_workers = 0;

    for (i = 0; i < opencl_state.num_platforms; i++)
    {
        int j;
        for (j = 0; j < opencl_state.num_devices[i]; j++)
        {
            struct worker_s *worker = &workers[num_workers];
            worker->opencl_state = &opencl_state;
            worker->dispatcher = &dispatcher;
            worker->platform_id = i;
            worker->device_id = j;
            worker->T = T;
            worker->h = h;
            worker->args = args;
            worker->num_args = num_args;
            worker->num_steps = num_steps;
            num_workers++;
        }
    }

    for (i = 0; i < num_workers; i++)
        pthread_create(&threads[i], NULL, worker_launcher, &workers[i]);
    
    for (i = 0; i < num_workers; i++)
        pthread_join(threads[i], NULL);

    release_opencl(&opencl_state);

    FILE *output = fopen(output_fname, "wt");
    fprintf(output, "finished");
    for (i = 0; i < DIM; i++)
        fprintf(output, ",pos%i", i);
    for (i = 0; i < DIM; i++)
        fprintf(output, ",dir%i", i);
    fprintf(output, "\n");

    for (i = 0; i < num_objects; i++)
    {
        if (finished[i])
            fprintf(output, "true");
        else
            fprintf(output, "false");
        int j;
        for (j = 0; j < DIM; j++)
            fprintf(output, ",%lf", (double)pos[DIM * i + j]);
        for (j = 0; j < DIM; j++)
            fprintf(output, ",%lf", (double)dir[DIM * i + j]);
        fprintf(output, "\n");
    }

    if (output_rays != NULL)
    {
        for (i = 0; i < num_objects; i++)
            fclose(output_rays[i]);
    }
    fclose(output);
    dispatcher_release(&dispatcher);
    return 0;
}
