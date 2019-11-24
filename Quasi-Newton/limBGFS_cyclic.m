function [xf, iter] = limBGFS_cyclic(f, x0, tol, maxiter, m)
    %UNTITLED2 Summary of this function goes here
    %   Detailed explanation goes here
    
    iter = 0;
    xf = x0;
    gk = grad(f, xf);
    n = length(gk);
    dk = -gk;

    S = zeros(n, m);
    G = S;
    startAt = 1;
    
    while norm(gk, 'inf') > tol && iter < maxiter
        
        [alpha, gnew] = lineSearch(f, xf, dk, gk);
        
        % Actualizar S, G
        S(:, startAt) = alpha*dk;
        G(:, startAt) = gnew - gk;
        
        
        %Calcular nuevo dk
        iter  = iter + 1;
        if iter < m
            dk = -calcHg(S, G, gnew, startAt, iter);
        else
            dk = -calcHg(S, G, gnew, startAt, m);
        end
        

        xf = xf + S(:, startAt);
        gk = gnew;
        startAt = mod(startAt, m) + 1;

    end
end

function [q] = calcHg(S, G, gnew, startAt, len)
    q = gnew;
    m = len;
    rhos = zeros(m, 1);
    alphas = zeros(m, 1);
    
    j = startAt;
    for i = 1:m
        
        rhos(i) = 1 / dot(S(:, j), G(:, j));
        alphas(i) = dot(S(:, j), q) * rhos(i);
        
        q = q - alphas(i)*G(:, j);
        j = mod(j, m) + 1;
    end


    delta = 1 / (rhos(1) * dot(G(:, startAt), G(:, startAt)));
    
    q = delta * q;
    

    j = j - 1;
    if j == 0
        j = m;
    end

    for i = m:-1:1
        beta = dot(G(:, j), q)*rhos(i);
        q = q + (alphas(i) - beta) * S(:, j);
        
        j = j - 1;
        if j == 0
            j = m;
        end
    end

end