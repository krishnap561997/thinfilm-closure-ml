%% ================================================================
%% CHANG WHITE-NOISE SIGNAL
%% ================================================================
function Fout = chang_noise_signal(t, MNOISE, fcut_noise, F0_noise)

    persistent noise_phase initialized old_MNOISE old_seed

    noise_seed = 1;

    if isempty(initialized) || old_MNOISE ~= MNOISE || old_seed ~= noise_seed

        fprintf('Initializing white noise phases...\n');

        rng(noise_seed, 'twister');

        noise_phase = 2*pi*rand(MNOISE,1);

        initialized = true;
        old_MNOISE = MNOISE;
        old_seed = noise_seed;
    end

    df = fcut_noise/MNOISE;

    F = 0.0;

    for k = 1:MNOISE
        fk = k*df;
        F = F + cos(2*pi*fk*t + noise_phase(k));
    end

    Fout = F0_noise*sqrt(2.0/MNOISE)*F;
end