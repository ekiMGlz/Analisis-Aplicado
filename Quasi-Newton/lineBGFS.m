function [xf, iter] = lineBGFS(f, x0, tol, maxiter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    iter = 0;
    xf = x0;
    gk = gradient(f, xf);
    n = length(gk);
    Hk = speye(n);
    
    while norm(gk, 'inf') > tol && iter < maxiter
        dk = -Hk*gk;
        
        [alpha, gnew] = encAlpha(f, xf, dk, gk);
        
        s = alpha*dk;
        gamma = gnew - gk;
        rho = 1/dot(gamma, s);
        
        xf = xf + s;
        gk = gnew;
        % BGFS Hk - Verificar con formula
        HkGamma = Hk*gamma*rho;
        sHkGammaT = s*HkGamma';
        Hk = -(sHkGammaT + sHkGammaT') + (rho*(dot(gamma, HkGamma) + 1)*s)*s';
        
        
        iter  = iter+1;
    end
end
