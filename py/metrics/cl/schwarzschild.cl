struct tensor_2 metric_tensor(const struct tensor_1 *pos, __global const real *args)
{
	real rs = args[0];

    struct tensor_2 g = {
        .covar = {true, true}
    };
    real r = pos->x[1];
    real theta = pos->x[2];
    real phi = pos->x[3];

    real k = 1 - rs / r;

    g.x[0][0] = k;                  // g_tt
    g.x[1][1] = -1/k;               // g_rr
    g.x[2][2] = -SQR(r);            // g_thth
    g.x[3][3] = -SQR(r*sin(theta)); //g_ff

    return g;
}

bool allowed_area(const struct tensor_1 *pos, __global const real *args)
{
   	real rs = args[0];
    real r = pos->x[1];

    if (fabs(r-rs) < 1e-4)
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
    real r = pos->x[1];
    real dr = dpos->x[1];
    if ((r > rs && r+dr <= rs) || (r < rs && r+dr >= rs))
        return false;
    return true;
}

struct tensor_2 contravariant_metric_tensor(const struct tensor_2 *g)
{
    return contravariant_metric_tensor_diagonal(g);
}
