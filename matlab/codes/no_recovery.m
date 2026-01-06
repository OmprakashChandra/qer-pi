function Fe = no_recovery(ket0L, ket1L, Kraus)
    ket0L = ket0L / norm(ket0L);
    ket1L = ket1L / norm(ket1L);

    d = length(ket0L);
    if length(ket1L) ~= d
        error('ket0L and ket1L must have the same dimension.');
    end

    rho = 0.5*(ket0L*ket0L' + ket1L*ket1L');

    Fe = 0;
    for k = 1:numel(Kraus)
        Ek = Kraus{k};
        if ~isequal(size(Ek), [d d])
            error('Each Kraus operator must be dim x dim.');
        end
        Fe = Fe + abs(trace(rho * Ek))^2;
    end

    Fe = real(Fe);
    %Fe = min(max(Fe, 0), 1);
end
