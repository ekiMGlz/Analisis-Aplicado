function [xf, iter] = lineDFP(f, x0, tol, maxiter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    iter = 0;
    xf = x0;
    gk = grad(f, xf);
    Hk = eye(length(gk));
    
    while norm(gk, 'inf') > tol && iter < maxiter
        dk = -Hk*gk;
        
        [alpha, gnew] = lineSearch(f, xf, dk, gk);
        
        s = alpha*dk;
        gamma = gnew - gk;
        irho = dot(gamma, s);
        
        xf = xf + s;
        gk = gnew;
        
        % DFP Hk
        HkGamma = Hk*gamma;
        Hk = Hk + (irho*s)*s' - HkGamma * (HkGamma' / dot(gamma, HkGamma));
        
        iter  = iter+1;
    end
end

