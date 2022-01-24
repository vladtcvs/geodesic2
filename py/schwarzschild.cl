struct tensor_2 metric_tensor(const struct tensor_1 *pos, __global const double *args)
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

bool allowed_area(const struct tensor_1 *pos, __global const double *args)
{
   	double rs = args[0];
    double r = pos->x[1];

    if (fabs(r-rs) < 1e-4)
        return false;
    return true;
}

bool allowed_delta(const struct tensor_1 *pos, const struct tensor_1 *dpos, __global const double *args)
{
    double rs = args[0];
    double r = pos->x[1];
    double dr = dpos->x[1];
    if ((r > rs && r+dr <= rs) || (r < rs && r+dr >= rs))
        return false;
    return true;
}
