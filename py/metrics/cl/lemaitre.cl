struct tensor_2 metric_tensor(const struct tensor_1 *pos, __global const real *args)
{
	real rs = args[0];

    struct tensor_2 g = {
        .covar = {true, true}
    };

    real tau = pos->x[0];
    real rho = pos->x[1];

    real theta = pos->x[2];
    real phi = pos->x[3];

    real r = powr(3.0/2.0 * (rho - tau), 2.0/3.0) * powr(rs, 1.0/3.0);

    g.x[0][0] = 1;                  // g_tau_tau
    g.x[1][1] = -rs/r;              // g_rho_rho
    g.x[2][2] = -SQR(r);            // g_thth
    g.x[3][3] = -SQR(r*sin(theta)); // g_ff

    return g;
}

bool allowed_area(const struct tensor_1 *pos, __global const real *args)
{
   	real rs = args[0];
    
    real tau = pos->x[0];
    real rho = pos->x[1];

    if (rho - tau < 1e-6)
        return false;
    return true;
}

bool allowed_delta(const struct tensor_1 *pos,
                   const struct tensor_1 *dir,
                   const struct tensor_1 *dpos,
                   const struct tensor_1 *ddir,
                   __global const real *args)
{
    real rs = args[0];

    real tau = pos->x[0];
    real rho = pos->x[1];

    real dtau = dpos->x[0];
    real drho = dpos->x[1];
    
    if (rho - tau + drho - dtau < 1e-4)
        return false;
    return true;
}

struct tensor_2 contravariant_metric_tensor(const struct tensor_2 *g)
{
    return contravariant_metric_tensor_diagonal(g);
}
