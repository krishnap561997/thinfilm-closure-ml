%% ================================================================
%% CONSTANT FREQUENCY SIGNAL
%% ================================================================
function Fout = inlet_signal(t, freq, F0)
    Fout = F0*sin(2*pi*freq*t);
end
