function [xf, iter] = limBGFS(f, x0, tol, maxiter, m)
    %UNTITLED2 Summary of this function goes here
    %   Detailed explanation goes here
    
    iter = 0;
    xf = x0;
    gk = grad(f, xf);
    n = length(gk);
    dk = -gk;

    S = zeros(n, m);
    G = S;
    
    
    while norm(gk, 'inf') > tol && iter < maxiter
        
        [alpha, gnew] = lineSearch(f, xf, dk, gk);
        
        % Actualizar S, G
        S(:, 2:end) = S(:, 1:end-1);
        S(:, 1) = alpha*dk;

        G(:, 2:end) = G(:, 1:end-1);
        G(:, 1) = gnew - gk;
        
        %Calcular nuevo dk
        iter  = iter + 1;
        if iter < m
            dk = -calcHg(S(:, 1:iter), G(:, 1:iter), gnew);
        else
            dk = -calcHg(S, G, gnew);
        end


        xf = xf + S(:, 1);
        gk = gnew;
        
    end
end

function [q] = calcHg(S, G, gnew)
    q = gnew;
    m = length(S(1, :));
    rhos = 1 ./ dot(S, G, 1);
    alphas = zeros(m, 1);
    
    for i = 1:m
        alphas(i) = dot(S(:, i), q) * rhos(i);
        q = q - alphas(i)*G(:, i);
    end

    delta = 1/(rhos(1) * dot(G(:, 1), G(:, 1)));
    q = delta * q;

    for i = m:-1:1
        beta = dot(G(:, i), q)*rhos(i);
        q = q + (alphas(i) - beta) * S(:, i);
    end

end