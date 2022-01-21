#define DIM 4

#define USE_SPHERIC   1
#define USE_DECART    0
#define USE_LEMAITRE  0

#define SQR(x) ((x)*(x))

struct tensor_1
{
    bool covar[1];
    double x[DIM];
};

struct tensor_2
{
    bool covar[2];
    double x[DIM][DIM];
};

struct tensor_3
{
    bool covar[3];
    double x[DIM][DIM][DIM];
};

/* Space description */

#if USE_SPHERIC
struct tensor_2 metric_tensor(const struct tensor_1 *pos, global const double *args)
{
	double rs = args[0];

    struct tensor_2 g = {
        .covar = {true, true}
    };
    double r = pos->x[1];
    double theta = pos->x[2];
    double phi = pos->x[3];

    double k = 1 - rs / r;

    g.x[0][0] = k;                  // g_tt
    g.x[1][1] = -1/k;               // g_rr
    g.x[2][2] = -SQR(r);            // g_thth
    g.x[3][3] = -SQR(r*sin(theta)); //g_ff

    return g;
}

#elif USE_DECART

struct tensor_2 metric_tensor(const struct tensor_1 *pos, global const double *args)
{
	double rs = args[0];

	struct tensor_2 g = {
        .covar = {true, true}
    };
	
    g.x[0][0] = 1;
    g.x[1][1] = -1;
    g.x[2][2] = -1;
    g.x[3][3] = -1;

    return g;
}

#elif USE_LEMAITRE

struct tensor_2 metric_tensor(const struct tensor_1 *pos, global const double *args)
{
	double rs = args[0];

	struct tensor_2 g = {
        .covar = {true, true}
    };
	
	double tau   = pos->x[0];
	double rho   = pos->x[1];
    double theta = pos->x[2];
    double phi   = pos->x[3];

	double r = pow(1.5*(rho-tau), 2.0/3.0) * pow(rs, 1.0/3.0);

    g.x[0][0] = 1;                  // g_tau_tau
    g.x[1][1] = -rs/r;              // g_rho_rho
    g.x[2][2] = -SQR(r);            // g_theta_theta
    g.x[3][3] = -SQR(r*sin(theta)); // g_phi_phi

    return g;
}

#endif

/**
 * Find contravariant metric tensor.
 * It is just inverted matrix `g`
 * @param metric metric tensor in covariant form
 * @return metric tensor in contravariant form
 */
struct tensor_2 contravariant_metric_tensor(const struct tensor_2 *metric)
{
	struct tensor_2 g = {
        .covar = {false, false}
    };

	// WARNING: Here we assume that metric is diagonal!
	// It is wrong in common case
	// TODO: handle common case

	int i;
	for (i = 0; i < DIM; i++)
		g.x[i][i] = 1 / metric->x[i][i];
	
	return g;
}

/**
 * Find derivative of metric tensor
 * @param pos position
 * @param args parametrs of metric
 * @return metric derivative
 */
struct tensor_3 metric_derivative_num(const struct tensor_1 *pos, global const double *args)
{
	int i, j, k;
	struct tensor_3 dg = {
        .covar = {true, true, true}
    };

	const double h = 1e-10;

	for (i = 0; i < DIM; i++)
	{
		struct tensor_1 pos1 = *pos, pos2 = *pos, pos4 = *pos, pos5 = *pos;
		pos1.x[i] -= 2*h;
		pos2.x[i] -= h;
		pos4.x[i] += h;
		pos5.x[i] += 2*h;

		struct tensor_2 g1 = metric_tensor(&pos1, args);
		struct tensor_2 g2 = metric_tensor(&pos2, args);
		struct tensor_2 g4 = metric_tensor(&pos4, args);
		struct tensor_2 g5 = metric_tensor(&pos5, args);

		for (j = 0; j < DIM; j++)
		for (k = 0; k < DIM; k++)
		{
			dg.x[i][j][k] = (g1.x[j][k] - 8*g2.x[j][k] + 8*g4.x[j][k] - g5.x[j][k]) / (12*h);
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
struct tensor_3 cristofel_symbol(const struct tensor_1 *pos, global const double *args)
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
struct tensor_1 geodesic_diff(const struct tensor_1 *pos, const struct tensor_1 *dir, global const double *args)
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
 */
void geodesic_calculation_step(struct tensor_1 *pos, struct tensor_1 *dir, double h, global const double *args)
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

    for (i = 0; i < DIM; i++)
    {
        pos->x[i] += (pos_k1.x[i] + pos_k2.x[i]*2 + pos_k3.x[i]*2 + pos_k4.x[i]) * h/6;
        dir->x[i] += (dir_k1.x[i] + dir_k2.x[i]*2 + dir_k3.x[i]*2 + dir_k4.x[i]) * h/6;
    }
}

/**
 * Iteration step. New values of `p` and `d` will be stored in place.
 * Runge-Kutta method is used.
 *
 * @param num amount of steps
 * @param pos current positions
 * @param dir current directions
 * @param h iteration step
 * @param args parameters of metric
 */
kernel void kernel_geodesic(int num, global double *pos, global double *dir, double h, global const double *args)
{
    int id = get_global_id(0);
    int i;

    struct tensor_1 cpos = {
        .covar = {false},
    };
    struct tensor_1 cdir = {
        .covar = {false},
    };
    
    for (i = 0; i < DIM; i++)
    {
        cpos.x[i] = pos[DIM*id + i];
        cdir.x[i] = dir[DIM*id + i];

		if (isnan(cpos.x[i]))
			return;
		if (isnan(cdir.x[i]))
			return;
    }

	for (i = 0; i < num; i++)
    	geodesic_calculation_step(&cpos, &cdir, h, args);

    for (i = 0; i < DIM; i++)
    {
        pos[DIM*id + i] = cpos.x[i];
        dir[DIM*id + i] = cdir.x[i];
    }
}
