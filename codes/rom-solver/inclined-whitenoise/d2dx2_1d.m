function fxx = d2dx2_1d(f, dx)

    f = f(:);
    N = length(f);
    fxx = zeros(N,1);

    fxx(2:N-1) = (f(3:N) - 2*f(2:N-1) + f(1:N-2))/(dx^2);

    fxx(1) = (2*f(1) - 5*f(2) + 4*f(3) - f(4))/(dx^2);
    fxx(N) = (2*f(N) - 5*f(N-1) + 4*f(N-2) - f(N-3))/(dx^2);
end