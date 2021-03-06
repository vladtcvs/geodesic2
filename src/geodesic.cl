#define DIM 4

#define SQR(x) ((x)*(x))

typedef double real;

#define diff_h 1e-6

struct tensor_1
{
    bool covar[1];
    real x[DIM];
};

struct tensor_2
{
    bool covar[2];
    real x[DIM][DIM];
};

struct tensor_3
{
    bool covar[3];
    real x[DIM][DIM][DIM];
};

/**
 * This functions must be added to code
 */
struct tensor_2 metric_tensor(const struct tensor_1 *pos, __global const real *args);
struct tensor_2 contravariant_metric_tensor(const struct tensor_2 *g);

bool allowed_area(const struct tensor_1 *pos, __global const real *args);
bool allowed_delta(const struct tensor_1 *pos,
                   const struct tensor_1 *dir,
                   const struct tensor_1 *dpos,
                   const struct tensor_1 *ddir,
                   __global const real *args);

void limit_dir(struct tensor_1 *dir)
{
    const real maxd = 1e2;
    int i;

    real md = 0;
    for (i = 0; i < DIM; i++)
    {
        if (fabs(dir->x[i]) > md)
            md = fabs(dir->x[i]);
    }

    if (md > maxd)
    {
        for (i = 0; i < DIM; i++)
        {
            dir->x[i] *= maxd / md;
        }
    }
}

/**
 * Find contravariant metric tensor for diagonal case
 * It is just inverted matrix `g`
 * @param g metric tensor in covariant form
 * @return metric tensor in contravariant form
 */
struct tensor_2 contravariant_metric_tensor_diagonal(const struct tensor_2 *g)
{
	struct tensor_2 ig = {
        .covar = {false, false}
    };

    int i;

    for (i = 0; i < DIM; i++)
        ig.x[i][i] = 1/g->x[i][i];
    
    return ig;
}

/**
 * Find derivative of metric tensor
 * @param pos position
 * @param args parametrs of metric
 * @return metric derivative
 */
struct tensor_3 metric_derivative_num(const struct tensor_1 *pos, __global const real *args)
{
	int i, j, k;
	struct tensor_3 dg = {
        .covar = {true, true, true}
    };

	for (i = 0; i < DIM; i++)
	{
		struct tensor_1 pos1 = *pos, pos2 = *pos, pos4 = *pos, pos5 = *pos;
		pos1.x[i] -= 2*diff_h;
		pos2.x[i] -= diff_h;
		pos4.x[i] += diff_h;
		pos5.x[i] += 2*diff_h;

		struct tensor_2 g1 = metric_tensor(&pos1, args);
		struct tensor_2 g2 = metric_tensor(&pos2, args);
		struct tensor_2 g4 = metric_tensor(&pos4, args);
		struct tensor_2 g5 = metric_tensor(&pos5, args);

		for (j = 0; j < DIM; j++)
		for (k = 0; k < DIM; k++)
		{
			dg.x[i][j][k] = (g1.x[j][k] - 8*g2.x[j][k] + 8*g4.x[j][k] - g5.x[j][k]) / (12*diff_h);
		}
	}
	return dg;
}

/**
 * Numerically calculate cristofel symbol
 * @param pos position
 * @param args parameters of metric
 * @return cristofel symbol
 */
struct tensor_3 cristofel_symbol(const struct tensor_1 *pos, __global const real *args)
{
	int i, m, k, l;

	struct tensor_2 metric = metric_tensor(pos, args);
	struct tensor_2 metric_contra = contravariant_metric_tensor(&metric);
	struct tensor_3 metric_derivative = metric_derivative_num(pos, args);

	struct tensor_3 G = {
        .covar = {false, true, true}
    };

	for (i = 0; i < DIM; i++)
	for (k = 0; k < DIM; k++)
	for (l = 0; l < DIM; l++)
	{
		G.x[i][k][l] = 0;
		for (m = 0; m < DIM; m++)
			G.x[i][k][l] += 0.5 * metric_contra.x[i][m]*(metric_derivative.x[l][m][k] + metric_derivative.x[k][m][l] - metric_derivative.x[m][k][l]);
	}

	return G;
}

/* End of space description */

/**
 * Calculate change of geodesic line direction while moving along it.
 * @param G Kristoffel symbol at current position
 * @param d current direction
 * @return change of direction after moving along d
 */
struct tensor_1 geodesic_diff_G(const struct tensor_3* G, const struct tensor_1 *d)
{
    struct tensor_1 d2 = {
        .covar = {false}
    };

    int i, j, k;
    for (k = 0; k < DIM; k++)
    {
        d2.x[k] = 0;
        for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++)
        {
            d2.x[k] -= G->x[k][i][j] * d->x[i] * d->x[j];
        }
    }
    return d2;
}

/**
 * Calculate change of geodesic line direction while moving along it.
 * @param pos current position
 * @param dir current direction
 * @param args parameters of metric
 * @return change of direction after moving along d
 */
struct tensor_1 geodesic_diff(const struct tensor_1 *pos, const struct tensor_1 *dir, __global const real *args)
{
    struct tensor_3 G = cristofel_symbol(pos, args);
    struct tensor_1 d = geodesic_diff_G(&G, dir);
    return d;
}

/**
 * Iteration step. New values of `p` and `d` will be stored in place.
 * Runge-Kutta method is used.
 *
 * @param pos current position
 * @param dir current direction
 * @param h iteration step
 * @param args parameters of metric
 * @return can we continue this geodesic
 */
bool geodesic_calculation_step(struct tensor_1 *pos, struct tensor_1 *dir, real h, __global const real *args)
{
    int i;
    struct tensor_1 dir_k1 = geodesic_diff(pos, dir, args);
    struct tensor_1 pos_k1 = *dir;

    struct tensor_1 pos_2 = *pos;
    struct tensor_1 dir_2 = *dir;
    for (i = 0; i < DIM; i++)
    {
        pos_2.x[i] += pos_k1.x[i] * h/2;
        dir_2.x[i] += dir_k1.x[i] * h/2;
    }

    struct tensor_1 dir_k2 = geodesic_diff(&pos_2, &dir_2, args);
    struct tensor_1 pos_k2 = dir_2;

    struct tensor_1 pos_3 = *pos;
    struct tensor_1 dir_3 = *dir;
    for (i = 0; i < DIM; i++)
    {
        pos_3.x[i] += pos_k2.x[i] * h/2;
        dir_3.x[i] += dir_k2.x[i] * h/2;
    }

    struct tensor_1 dir_k3 = geodesic_diff(&pos_3, &dir_3, args);
    struct tensor_1 pos_k3 = dir_3;
    
    struct tensor_1 pos_4 = *pos;
    struct tensor_1 dir_4 = *dir;
    for (i = 0; i < DIM; i++)
    {
        pos_4.x[i] += pos_k3.x[i] * h;
        dir_4.x[i] += dir_k3.x[i] * h;
    }

    struct tensor_1 dir_k4 = geodesic_diff(&pos_4, &dir_4, args);
    struct tensor_1 pos_k4 = dir_4;

    struct tensor_1 delta_pos;
    struct tensor_1 delta_dir;

    bool bad = false;

    for (i = 0; i < DIM; i++)
    {
        delta_pos.x[i] = (pos_k1.x[i] + pos_k2.x[i]*2 + pos_k3.x[i]*2 + pos_k4.x[i]) * h/6;
        delta_dir.x[i] = (dir_k1.x[i] + dir_k2.x[i]*2 + dir_k3.x[i]*2 + dir_k4.x[i]) * h/6;
        if (isnan(delta_pos.x[i]) || isinf(delta_pos.x[i]) || isnan(delta_dir.x[i]) || isinf(delta_dir.x[i]))
            return false;
    }

    if (!allowed_delta(pos, dir, &delta_pos, &delta_dir, args))
        return false;

    for (i = 0; i < DIM; i++)
    {
        pos->x[i] += delta_pos.x[i];
        dir->x[i] += delta_dir.x[i];
    }

    return true;
}

/**
 * Iteration step. New values of `p` and `d` will be stored in place.
 * Runge-Kutta method is used.
 *
 * @param num amount of steps
 * @param pos current positions
 * @param dir current directions
 * @param finished status of each geodesic
 * @param h iteration step
 * @param args parameters of metric
 */
kernel void kernel_geodesic(int num, __global real *pos, __global real *dir, __global int *finished, real h, __global const real *args)
{
    int id = get_global_id(0);
    int i, j;

    struct tensor_1 cpos = {
        .covar = {false},
    };
    struct tensor_1 cdir = {
        .covar = {false},
    };
    
    if (finished[id] == 1)
        return;

    for (i = 0; i < DIM; i++)
    {
        cpos.x[i] = pos[DIM*id + i];
        cdir.x[i] = dir[DIM*id + i];
    }

    bool bad_ray = false;
	for (i = 0; i < num; i++)
    {
        if (!allowed_area(&cpos, args))
        {
            bad_ray = true;
            finished[id] = 1;
            break;
        }

        limit_dir(&cdir);

    	if (!geodesic_calculation_step(&cpos, &cdir, h, args))
        {
            finished[id] = 1;
            break;
        }
    }

    if (!bad_ray)
    {
        for (i = 0; i < DIM; i++)
        {
            pos[DIM*id + i] = cpos.x[i];
            dir[DIM*id + i] = cdir.x[i];
        }
    }
}
