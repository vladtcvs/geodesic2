#include <math.h>
#include <stdio.h>

typedef double real;

const real e = 2.71828;

real euler(real x)
{
    return x * exp(x);
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

int main(void)
{
	real x = -0.98;
	real y = euler(x);

	real x2 = W0_newton(y);
	printf("x=%lf, y=%lf, x2=%lf\n", x, y, x2);
	return 0;
}
