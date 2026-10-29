function fxxx = d3dx3_1d(f, dx)

    f = f(:);
    N = length(f);
    fxxx = zeros(N,1);

    fxxx(3:N-2) = ...
        (f(5:N) - 2*f(4:N-1) + 2*f(2:N-3) - f(1:N-4))/(2*dx^3);

    fxxx(1) = fxxx(3);
    fxxx(2) = fxxx(3);
    fxxx(N-1) = fxxx(N-2);
    fxxx(N) = fxxx(N-2);
end