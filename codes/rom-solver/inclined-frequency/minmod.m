%% ================================================================
%% MINMOD LIMITER
%% ================================================================
function s = minmod(a,b)

    s = zeros(size(a));
    idx = (a.*b) > 0;
    s(idx) = sign(a(idx)).*min(abs(a(idx)),abs(b(idx)));

end