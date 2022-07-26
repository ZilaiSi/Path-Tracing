function xi = GenGammaUpdate1D(z,r,beta)

% The function calculates the updates of the xi-variables corresponding to
% the z-values given in the input vector z using the generalized gamma model
% with parameters (r,beta)
% NOTE: This version assumes that all entries of the unknowns have their
% own individual prior variance as in 1D problems.
% FOR THE FUTURE (not done here):
% To avoid excessively fine discretization of the z-axis, in particular for
% the multidomensional problems, an approximate discretization is made
%--------------------------------------------------------------------------

n = length(z);

eta = r*beta - 3/2;
if eta/r < 0
    % There is a problem with parameters
    disp('Check the parameter values (r,beta)');
    return
end

Phi0     = (eta/r)^(1/r); % Initial value
PhiPrime = @(z,phi) 2*z*phi/(2*r^2*phi^(r+1)+z^2);


[z_sort,I_sort] = sort(abs(z),'ascend');
% Removing repeated values
[zz,I1,I2] = unique(z_sort);
flag = 0;
if zz(1) > 0
    % Zero added
    flag = 1;
    zz = [0;zz];
end


[~,xi] = ode45(PhiPrime,zz,Phi0);
if flag == 1
    % remove the first value
    xi = xi(2:end);
end
xi2 = xi(I2);
E = speye(n);
E = E(I_sort,:);
xi = E'*xi2;





