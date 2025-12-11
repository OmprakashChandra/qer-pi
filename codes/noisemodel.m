function model = noisemodel(noisetype, param, dim)
    %function to generate different noise models. For now only global symmetric noise is supported.
    %TODO add i) Amplitude damping, ii) Local depolarising noise
    switch noisetype
        case "globalsymmetric"
            model = globalsymmetric(param, dim); 
        otherwise
            error('Unsupported noise model type. The noise models supported at the moment are: globalsymmetric.');   
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
end 
