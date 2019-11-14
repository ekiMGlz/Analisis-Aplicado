function [alpha, gnew] = encAlpha(f, xk, dk, gk)
    
    alpha = 1;
    c1 = 1e-4;
    c2 = 0.99;
    gTd = dot(gk, dk);
    fk = f(xk);
    
    while f(xk + alpha*dk) > fk + alpha*c1*gTd
        alpha = alpha * 0.5;
    end
    
    gnew = grad(f, xk + alpha*dk);
    
    %assert(dot(gnew, dk) >= c2*gTd, "No se satisface W2")
    
    
end