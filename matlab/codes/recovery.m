function reshaped_eigenvectors = recovery(X, logical0, logical1)
    % RECOVERY Converts the Choi-like recovery operator X into Kraus operators in Dicke subspace.
    %
    % Inputs:
    %   X        - (d_l*d × d_l*d) recovery operator from SDP (X = X1 + 1i*X2)
    %   logical0 - (d × 1) logical-0 codeword in Dicke subspace
    %   logical1 - (d × 1) logical-1 codeword in Dicke subspace
    %
    % Output:
    %   reshaped_eigenvectors - cell array of (d × d) Kraus operators in Dicke subspace

    cutoff = 1e-4;  % numerical cutoff for eigenvalues

    % Convert X to full matrix if sparse
    if issparse(X)
        X = full(X);
    end

    % Dimensions
    d   = length(logical0);  % physical (Dicke) subspace dimension
    d_l = 2;                 % logical subspace dimension

    % Define encoder: (d × d_l) isometry mapping logical to physical space
    zero    = [1; 0];
    one     = [0; 1];
    Encoder = logical0 * zero' + logical1 * one';  % d × d_l

    % Use Schur decomposition (handles degenerate eigenvalues better than eig)
    [eigenvectors, eigen_matrix] = schur(X);
    eigen_values = diag(eigen_matrix);

    % Find eigenvalues above cutoff (use real part for numerical stability)
    valid_indices = find(real(eigen_values) > cutoff);

    % Initialize cell array for Kraus operators
    reshaped_eigenvectors = cell(1, length(valid_indices));

    for i = 1:length(valid_indices)
        idx    = valid_indices(i);
        vec    = eigenvectors(:, idx);
        lambda = eigen_values(idx);

        % Reshape: d_l*d vector → d_l × d matrix (column-major)
        R = reshape(vec, [d_l, d]);

        % Scale by sqrt(eigenvalue) and apply encoder: (d × d_l) * (d_l × d) = d × d
        reshaped_eigenvectors{i} = sqrt(lambda) * Encoder * R;
    end
end