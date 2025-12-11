function x = TrX(p, sys, dim)
    % TRX Computes the partial trace over specified subsystems.
    % p   : state vector or density matrix.
    % sys : subsystems to trace out.
    % dim : dimensions of the subsystems of p.
    
    n = length(dim);
    rdim = dim(end:-1:1);
    keep = 1:n;
    keep(sys) = [];
    dimtrace = prod(dim(sys));
    dimkeep = length(p)/dimtrace;
    
    if any(size(p) == 1)
        if size(p,1) == 1
            p = p';
        end
        perm = n+1 - [keep(end:-1:1), sys];
        x = reshape(permute(reshape(p, rdim), perm), [dimkeep, dimtrace]);
        x = x*x';
    else 
        perm = n+1 - [keep(end:-1:1), keep(end:-1:1)-n, sys, sys-n];
        x = reshape(permute(reshape(p, [rdim, rdim]), perm), [dimkeep, dimkeep, dimtrace^2]);
        x = sum(x(:, :, 1:dimtrace+1:dimtrace^2), 3);
    end
end 