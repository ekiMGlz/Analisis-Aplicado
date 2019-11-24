function [alpha, gnew] = lineSearch(f, xk, dk, gk)

    c1 = 1e-4;
    c2 = 0.99;
    maxIter = 1000;
    i = 1;

    alpha_0 = 0;
    alpha_max = 4;

    gTd = dot(gk, dk);
    fk = f(xk);
    phi_0 = fk;


    while i <= maxIter
        alpha = (alpha_0+alpha_max)*0.5;
        phi_i = f(xk + alpha*dk);
        gnew = grad(f, xk + alpha*dk);
        d_phi = dot(gnew, dk);
        
        if ( phi_i > fk + alpha*c1*gTd ) || ( phi_i >= phi_0 )
            [alpha, gnew] = zoom(alpha_0, alpha, f, xk, dk, fk, gTd);
            break
        elseif (abs(d_phi) <= -c2*gTd)
            break
        elseif d_phi >= 0
            [alpha, gnew] = zoom(alpha, alpha_0, f, xk, dk, fk, gTd);
            break
        else
            alpha_0 = alpha;
            phi_0 = phi_i;
            i = i + 1;
        end

    end

    assert(abs(dot(gnew, dk)) <= -c2*gTd, "No se satisface W2")
end

function [alpha, gnew] = zoom(a_low, a_high, f, xk, dk, fk, gTd)
    
    c1 = 1e-4;
    c2 = 0.99;
    
    maxIter = 1000;
    i = 1;

    phi_low = f(xk + a_low*dk);

    while (i <= maxIter)
        alpha = (a_low + a_high)*0.5;
        phi_i = f(xk + alpha*dk);
        gnew = grad(f, xk + alpha*dk);
        d_phi = dot(gnew, dk);

        if ( phi_i > fk + alpha*c1*gTd ) || ( phi_i >= phi_low )
            a_high = alpha;
            i = i + 1;
        elseif (abs(d_phi) <= -c2*gTd)
            break
        else
            if (d_phi*(a_high - a_low) >= 0)
                a_high = a_low;
            end
            a_low = alpha;
            phi_low = phi_i;
            i = i + 1;
        end
    end

    assert(abs(d_phi) <= -c2*gTd, "No se satisface W2")
end