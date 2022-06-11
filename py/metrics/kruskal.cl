real euler(real x)
{
    return x * exp(x);
}

real euler_dot(real x)
{
    return (1+x)*exp(x);
}

real newton_iteration(real x, real y)
{
    // x' = x - f(x) / f'(x)
    // f(x) = euler(x) - y
    // x' = x - (x*exp(x) - y) / ((x+1)*exp(x))
    real expx = exp(x);
    return x - (x*expx - y) / ((x+1)*expx);
}

real W0_newton(real y)
{
    const real e = 2.718281828459045;
    if (y <= -1)
        return -1/e;
    real x = 0;
    real xn = x;
    const real dx = 1e-8;

    do
    {
        x = xn;
        xn = newton_iteration(x, y);
        if (xn <= -1)
        {
            xn = (xn+x)/2;
        }
    } while (fabs(x - xn) > dx);

    return xn;
}

real W0_binary(real y)
{
    const real e = 2.718281828459045;
    if (y <= -1)
        return -1/e;
    
    real dx = 1e-8;
    
    real x1 = -1;
    real x2 = 100000;

    real e1 = euler(x1);
    real e2 = euler(x2);

    while (x2 - x1 > dx)
    {
        real xv = (x1 + x2)/2;
        real ev = euler(xv);

        if (ev > y)
        {
            x2 = xv;
            e2 = ev;
        }
        else
        {
            x1 = xv;
            e1 = ev;
        }
    }

    return (x1 + x2)/2;
}

real W0(real y)
{
    return W0_newton(y);
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
