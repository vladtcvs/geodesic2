#include <dispatcher.h>

#define min(a,b) ((a)<(b)?(a):(b))

void dispatcher_init(struct dispatcher_s *dispatcher, real *pos, real *dir, cl_int *finished, FILE **output, size_t num_objects)
{
    dispatcher->output = output;
    dispatcher->finished = finished;
    dispatcher->pos = pos;
    dispatcher->dir = dir;
    dispatcher->num_completed = 0;
    dispatcher->num_objects = num_objects;
    dispatcher->max_per_block = 256;
}

size_t dispatcher_get_next_block(struct dispatcher_s *dispatcher,
                                 real **pos,
                                 real **dir,
                                 cl_int **finished,
                                 FILE ***output,
                                 int amount)
{
    if (amount <= 0)
    {
        amount = dispatcher->max_per_block;
    }

    size_t num = min(dispatcher->num_objects - dispatcher->num_completed, amount);
    if (num > 0)
    {
        *pos = &(dispatcher->pos[dispatcher->num_completed*DIM]);
        *dir = &(dispatcher->dir[dispatcher->num_completed*DIM]);
        if (dispatcher->output != NULL)
        {
            *output = &(dispatcher->output[dispatcher->num_completed]);
        }
        else
        {
            *output = NULL;
        }
    
        *finished = &(dispatcher->finished[dispatcher->num_completed]);
        dispatcher->num_completed += num;
    }
    return num;
}

bool dispatcher_has_data(const struct dispatcher_s *dispatcher)
{
    return (dispatcher->num_objects > dispatcher->num_completed);
}
