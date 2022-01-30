real euler(real x)
{
    return x * exp(x);
}

real W0(real x)
{
    if (x < -1)
        return -1;
    
    real dy = 1e-8;
    
    real y1 = -1;
    real y2 = 100000;

    real e1 = euler(y1);
    real e2 = euler(y2);

    while (y2 - y1 > dy)
    {
        real yv = (y1 + y2)/2;
        real ev = euler(yv);

        if (ev > x)
        {
            y2 = yv;
            e2 = ev;
        }
        else
        {
            y1 = yv;
            e1 = ev;
        }
    }

    return (y1 + y2)/2;
}

real radius_relative(real T, real X)
{
    // T**2 - X**2 = (1 - r/rs) * exp(r/rs)
    return 1 + W0((X*X - T*T) / 2.71828182846);
}

struct tensor_2 metric_tensor(const struct tensor_1 *pos, __global const real *args)
{
	real rs = args[0];

    struct tensor_2 g = {
        .covar = {true, true}
    };

    real T = pos->x[0];
    real X = pos->x[1];

    real theta = pos->x[2];
    real phi = pos->x[3];

    real r = radius_relative(T, X);

    real k = 4 * rs*rs / r * exp(-r);

    g.x[0][0] = -k;                // g_TT
    g.x[1][1] = k;                 // g_XX
    g.x[2][2] = SQR(r*rs);            // g_thth
    g.x[3][3] = SQR(r*rs*sin(theta)); //g_ff

    return g;
}

bool allowed_area(const struct tensor_1 *pos, __global const real *args)
{
   	real rs = args[0];

    real T = pos->x[0];
    real X = pos->x[1];
    
    if (fabs(T) > 1000)
    {
        return false;
    }

    real r = radius_relative(T, X);
    if (r < 1e-6)
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

    real T = pos->x[0];
    real X = pos->x[1];

    if (fabs(X) > fabs(T))
        return true;

    if (fabs(ddir->x[0]) > 100 || fabs(ddir->x[1]) > 100)
        return false;

    return true;
}

struct tensor_2 contravariant_metric_tensor(const struct tensor_2 *g)
{
    return contravariant_metric_tensor_diagonal(g);
}
