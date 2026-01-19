function model = noisemodel(noisetype, param, dim)
    %function to generate different noise models. For now only global symmetric noise is supported.
    %TODO add  i) Local depolarising noise
    %Inputs:
    % noisetype : string specifying the noise model type.
    % param     : parameter specifying the strength of the noise.
    % dim       : dimension of the Dicke subspace which will be N+1. Where N is the number of qubits.
    %Outputs: 
    % model     : cell array of Kraus operators representing the noise model.
    switch noisetype
        case "globalsymmetric"
            model = globalsymmetric(param, dim); 
        case "tracepreserving"
            model = tracepreserving(param, dim);
        case "collectdeph"
            model = collective_dephasing(param, dim);  
        case "collectdeph_new"
            model = collective_dephasing_new(param, dim);    
        case "localampdamp"
            model = local_ampdamp(param, dim);      
        otherwise
            error('Unsupported noise model type. The noise models supported at the moment are: "globalsymmetric", "tracepreserving" and "ampdamp".');   
    end  
    
    function model = globalsymmetric(param, dim)
        persistent cache_dim cache_Ex cache_Ey cache_Ez cache_s
        if isempty(cache_dim) || cache_dim ~= dim 
            n = dim - 1;
            s = n / 2;
            Esm = zeros(dim);
            % Generate the Jx, Jy and Jz operators in the Dicke subspace. Note that this is a
            %  generalisation of the depolarising channel to higher dimensions.
            for j = -s:s-1
                Esm(j+s+1, j+s+2) = sqrt(s*(s+1) - j*(j+1));
            end
            Esp = Esm';
            Ex = (Esm + Esp) / 2;
            Ey = (Esm - Esp) / (2*1i);
            Ez = (Esp*Esm - Esm*Esp) / 2;
            cache_dim = dim;
            cache_Ex = Ex;
            cache_Ey = Ey;
            cache_Ez = Ez;
            cache_s = s; 
        end   
        p = param; % noise parameter
        E0 = ((1 - 0.5*(p*cache_s*(cache_s+1)))) * eye(dim);
        E1 = sqrt(p) * cache_Ex;
        E2 = sqrt(p) * cache_Ey;
        E3 = sqrt(p) * cache_Ez;
        model = {E0, E1, E2, E3};            
    end 

    function model = tracepreserving(param, dim)
            % Trace-preserving global symmetric noise model
            persistent cache_dim cache_Ex cache_Ey cache_Ez cache_s
            if isempty(cache_dim) || cache_dim ~= dim 
                n = dim - 1;
                s = n / 2;
                Esm = zeros(dim);
                for j = -s:s-1
                    Esm(j+s+1, j+s+2) = sqrt(s*(s+1) - j*(j+1));
                end
                Esp = Esm';
                Ex = (Esm + Esp) / 2;
                Ey = (Esm - Esp) / (2*1i);
                Ez = (Esp*Esm - Esm*Esp) / 2;
                cache_dim = dim;
                cache_Ex = Ex;
                cache_Ey = Ey;
                cache_Ez = Ez;
                cache_s = s;    
            end 
            p = param; % noise parameter
            E0 = sqrt(1 - p * cache_s *(cache_s + 1)) * eye(cache_dim);
            E1 = sqrt(p) * cache_Ex;
            E2 = sqrt(p) * cache_Ey;
            E3 = sqrt(p) * cache_Ez;
            model = {E0, E1, E2, E3};
        end 

    function K = collective_dephasing(param, dim)

        if dim < 1 || floor(dim) ~= dim
            error('dim must be a positive integer.');
        end

        N = dim - 1;
        J = N / 2;

        if J > 0
            maxParam = 1 / (2 * J * (J + 1));
            if param < 0 || param > maxParam
                error('param must be in [0, 1/(2*J*(J+1))] = [0, %.3g] for N = %d.', maxParam, N);
            end
        else
            if param ~= 0
                error('For dim = 1 (N = 0), param must be 0.');
            end
        end

        Sz = zeros(dim);
        for k = 0:N
            m = k - J;
            Sz(k+1, k+1) = m;
        end

        Sm = zeros(dim);
        for k = 1:N
            amp = sqrt(k * (N - k + 1));
            Sm(k, k+1) = amp;
        end

        Sp = Sm';

        H = Sz^2 + Sp*Sm + Sm*Sp;
        I = eye(dim);

        K0 = sqrtm(I - param * H);
        K1 = sqrt(param) * Sz;
        K2 = sqrt(param) * Sm;
        K3 = sqrt(param) * Sp;

        K = {K0, K1, K2, K3};

    end 

    function K = collective_dephasing_new(param, dim)

        N = dim - 1;
        J = N / 2;

        if param < 0 || param > 1/(2 * J * (J + 1))
            error('param must be in [0, 1/(2*J*(J+1))] for N = %d.', N);
        end

        Jz = zeros(dim);
        for k = 0:N
            m = k - J;
            Jz(k+1, k+1) = m;
        end

        Jp = zeros(dim);
        for k = 0:N-1
            amp = sqrt((k+1) * (N - k));
            Jp(k+2, k+1) = amp;
        end
        Jm = Jp';

        K1 = sqrt(param)      * Jp;
        K2 = sqrt(param)      * Jm;
        K3 = sqrt(2 * param)  * Jz;
        K4 = sqrt(1 - 2*param*J*(J+1)) * eye(dim);

        K = {K1, K2, K3, K4};

    end 

    function Kraus = local_ampdamp(param, dim)
        gamma = param; 
        % Generate the Kraus operators for local amplitude damping noise on N qubits.
        if ~(isscalar(dim) && dim == round(dim) && dim >= 2)
            error('dim must be an integer >= 2 (and equal to 2^N).');
        end
        if ~(isscalar(gamma) && isreal(gamma) && gamma >= 0 && gamma <= 1)
            error('gamma must be a real scalar in [0,1].');
        end
        
        N = dim; %number of qubits  
        %N = round(log2(dim));
        %if 2^N ~= dim
        %    error('dim must be exactly a power of 2 (dim = 2^N).');
        %end

        % ---- single-qubit amplitude damping Kraus ops ----
        E0 = [1 0; 0 sqrt(1-gamma)];
        E1 = [0 sqrt(gamma); 0 0];
        E  = {E0, E1};  % E{1}=E0, E{2}=E1

        % ---- build N-qubit Kraus ops ----
        numK = 2^N;
        Kraus = cell(1, numK);

        for s = 0:(numK-1)
            bits = dec2bin(s, N) - '0'; % length-N, leftmost bit is MSB
            K = 1;
            for q = 1:N
                K = kron(K, E{bits(q)+1});
            end
            Kraus{s+1} = K;
        end
    end
end

