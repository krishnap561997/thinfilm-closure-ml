function fx = ddx_1d(f, dx)

    f = f(:);
    N = length(f);
    fx = zeros(N,1);

    fx(2:N-1) = (f(3:N) - f(1:N-2))/(2*dx);

    fx(1) = (-3*f(1) + 4*f(2) - f(3))/(2*dx);
    fx(N) = (3*f(N) - 4*f(N-1) + f(N-2))/(2*dx);
end