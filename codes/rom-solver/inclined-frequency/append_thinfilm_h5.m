function append_thinfilm_h5(fname, x, t, h, q, r,  dhdt, dqdt, ...
                            F_inlet, q_inlet, params)

    persistent is_initialized save_index

    h = h(:);
    q = q(:);
    qh = q./h;
    r = r(:);
    x = x(:);

    dhdt = dhdt(:);
    dqdt = dqdt(:);

    Nx = length(x);
    dx = x(2) - x(1);

    %% Spatial derivatives
    hx   = ddx_1d(h, dx);
    hxx  = d2dx2_1d(h, dx);
    hxxx = d3dx3_1d(h, dx);

    qx   = ddx_1d(q, dx);
    qxx  = d2dx2_1d(q, dx);
    
    qhx   = ddx_1d(qh, dx);
    qhxx  = d2dx2_1d(qh, dx);

    %% Time derivatives from previous solver timestep
    h_t = dhdt;
    q_t = dqdt;

    %% Initialize file
    if isempty(is_initialized) || ~isfile(fname)

        if isfile(fname)
            delete(fname);
        end

        save_index = 1;

        h5create(fname, '/x', size(x));
        h5write(fname, '/x', x);

        h5create(fname, '/time', [Inf 1], 'ChunkSize', [100 1]);

        h5create(fname, '/h', [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/q', [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/qh', [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/r', [Inf Nx], 'ChunkSize', [1 Nx]);

        h5create(fname, '/h_x',   [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/h_xx',  [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/h_xxx', [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/h_t',   [Inf Nx], 'ChunkSize', [1 Nx]);

        h5create(fname, '/q_x',   [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/q_xx',  [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/q_t',   [Inf Nx], 'ChunkSize', [1 Nx]);

        h5create(fname, '/qh_x',   [Inf Nx], 'ChunkSize', [1 Nx]);
        h5create(fname, '/qh_xx',  [Inf Nx], 'ChunkSize', [1 Nx]);

        h5create(fname, '/forcing/F_inlet', [Inf 1], 'ChunkSize', [100 1]);
        h5create(fname, '/forcing/q_inlet', [Inf 1], 'ChunkSize', [100 1]);

        names = fieldnames(params);
        for k = 1:numel(names)
            h5writeatt(fname, '/', names{k}, params.(names{k}));
        end

        is_initialized = true;

    else
        save_index = save_index + 1;
    end

    %% Write data
    h5write(fname, '/time', t, [save_index 1], [1 1]);

    h5write(fname, '/h', h.', [save_index 1], [1 Nx]);
    h5write(fname, '/q', q.', [save_index 1], [1 Nx]);
    h5write(fname, '/qh', qh.', [save_index 1], [1 Nx]);
    h5write(fname, '/r', r.', [save_index 1], [1 Nx]);

    h5write(fname, '/h_x',   hx.',   [save_index 1], [1 Nx]);
    h5write(fname, '/h_xx',  hxx.',  [save_index 1], [1 Nx]);
    h5write(fname, '/h_xxx', hxxx.', [save_index 1], [1 Nx]);
    h5write(fname, '/h_t',   h_t.',  [save_index 1], [1 Nx]);

    h5write(fname, '/q_x',   qx.',   [save_index 1], [1 Nx]);
    h5write(fname, '/q_xx',  qxx.',  [save_index 1], [1 Nx]);
    h5write(fname, '/q_t',   q_t.',  [save_index 1], [1 Nx]);

    h5write(fname, '/qh_x',   qhx.',   [save_index 1], [1 Nx]);
    h5write(fname, '/qh_xx',  qhxx.',  [save_index 1], [1 Nx]);


    h5write(fname, '/forcing/F_inlet', F_inlet, [save_index 1], [1 1]);
    h5write(fname, '/forcing/q_inlet', q_inlet, [save_index 1], [1 1]);
end