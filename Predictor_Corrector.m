function [X,Theta,rs,etas,theta_0s] = Predictor_Corrector(A,b,x0,theta0,r_range,eta_range,theta_0_range,N,method,steps_num)
% Predictor_Corrector: Predictor-Corrector Method

%%% Outputs
% X: restored signal
% Theta: variane
% rs: parameters r
% eta: parameters eta
% theta_0s: parameters theta_0
% stepsize: Newton stepsize
% converges: whether cgs converges
% sinvalue: smallest singular value of Hessian

%%% Inputs
% A: forward model A
% b: blurred signal
% x0: initial solution on the starting point of hyperparameters curve
% r_range: range of r
% eta_range: range of eta
% theta_0: range of theta_0
% N: number of equidistant time steps
% method: PN: prediction-Newton correction
%         IAS: path tracing IAS
%         IIAS: path tracing Inexact IAS
%         Ptheta: prediction-theta correction
%         PIAS: Prediction-IAS correction
%
% steps_num: number of interations


%% Get dimensions
n = length(x0);

%% fix r, eta and theta_0 trajectories
rs = linspace(r_range(1),r_range(end),N + 1);
etas = linspace(eta_range(1),eta_range(end),N + 1);
theta_0s = linspace(theta_0_range(1),theta_0_range(end),N + 1);

dt = 1/N; % time step

%% get rate of change in hyperparameters
drdt = rs(end) - rs(1);
detadt = etas(end) - etas(1);
dtheta_0dt = theta_0s(end) - theta_0s(1);

%% Preallocate
X = nan([n,N+1]);
Theta = nan([n,N+1]);
Phi = nan([n,N+1]);
%% initialize
x = x0;
theta = theta0;
phi = log(theta0); % change variable

X(:,1) = x;
Theta(:,1) = theta;
Phi(:,1) = phi;

% initialize preconditioner
preconditioner.mode = 'compute';
preconditioner.dt = dt;

y_old = nan; % initialization of ODE
%% iterate
for j = 2:N+1
    if (method == "PN")
        r = rs(j-1);
        eta = etas(j-1);
        theta_0 = theta_0s(j-1);
        % Prediction Step
        [dxdt,dphidt,y,preconditioner] = Compute_Derivatives(x,phi,r,theta_0,drdt,detadt,dtheta_0dt,A,preconditioner,y_old);
        y_old = y; % use solution of last iteration
        x = x + dxdt*dt;
        phi = phi + dphidt*dt;
        theta = exp(phi);
        %% Correction Step
        % Update hyperparameters
        r = rs(j);
        eta = etas(j);
        theta_0 = theta_0s(j);
        discrepancy = A*x - b;
        g = [A'*discrepancy + theta.^(-1).*x; -(1/2)*theta.^(-1).*x.^2 - eta + theta_0^(-r)*r*theta.^r];
        count = 1;
        while ( count <= steps_num)
            count = count+1;
            [step,preconditioner] = newtonstep(x,phi,r,eta,A,b,preconditioner,theta_0);
            discrepancy = A*x - b;
            g = [A'*discrepancy + theta.^(-1).*x; -(1/2)*theta.^(-1).*x.^2 - eta + theta_0^(-r)*r*theta.^r];
            if (step'*g>0)
                step = -g;
            end
            t = 1;
            alpha = 0.9;
            beta = 0.9;

            cost = @(A,b,x,theta,eta,r,theta_0) 1/2*sum((A*x-b).^(2)) + 1/2*sum(theta.^(-1).*(x.^2))-eta*sum((log(theta/theta_0)))+sum((theta/theta_0).^(r));

            while (cost(A,b,x+step(1:n)*t,exp(phi+step(n+1:2*n)*t),eta,r,theta_0) > cost(A,b,x,theta,eta,r,theta_0 ) + alpha*t*step'*g)
                t = t*beta;
                if t <= eps
                    break
                end
            end
            x = x + step((1:n))*t;
            phi = phi + step((n+1:2*n))*t;
            theta = exp(phi);
        end
    elseif (method == "IAS")
        r = rs(j);  %Update hyperparameters
        eta = etas(j);
        theta_0 = theta_0s(j);
        count = 1;
        while (count <= steps_num)
            M   = [A;spdiags(1./sqrt(theta),0,n,n)];
            rhs = [b;zeros(n,1)];
            shape_param = (3/2 + eta)/r;
            x = M\rhs;   % Update with exact IAS
            zeta = x./sqrt(theta_0);
            xi = GenGammaUpdate1D(zeta,r,shape_param);
            theta = theta_0.*xi;
            phi = log(theta);
            count = count+ 1;
        end

    elseif (method == "IIAS")

        r_gamma = rs(j);  %Update hyperparameters
        eta = etas(j);
        theta_0 = theta_0s(j);
        shape_param = (3/2 + eta)/r_gamma;
        s = 0.05:0.01:0.95;
        m = length(s);
        maxit_outer = 1;
        maxit_inner = m;
        F_a = zeros(1,maxit_outer);
        R_a = zeros(1,maxit_outer);
        G_a = zeros(1,maxit_outer);

        ObjFcn = @(x,th) 1/2*norm(A*x-b)^2 + 1/2*sum((x.^2)./th) + sum((th./theta_0).^r_gamma) ...
            - (r_gamma*shape_param - 3/2)*sum(log(th./theta_0));

        tol = 1e-2;
        DF_a = inf;
        Dth_a = inf;
        count = 0;

        while (DF_a>tol) && (count<maxit_outer) && Dth_a > tol
            count = 1;
            w = zeros(n,1);  % Initialize
            AD = A*diag(sqrt(theta));
            d = b - AD*w;
            r = AD'*d;
            p = r;
            aux0 = r'*r;
            h = 0;
            iterate = 'yes';
            G_prev = inf;
            discr = sqrt(m);
            while (h <= maxit_inner) && strcmp(iterate,'yes')
                h = h + 1;
                d_old = d;
                y = AD*p;
                alpha = aux0/(y'*y);
                w_old = w;
                w = w + alpha*p;       % New update
                d = d - alpha*y;
                % Checking the discrepancy condition
                G = norm(AD*w - b)^2 + norm(w)^2;
                if norm(d) < discr || norm(d-d_old) < 1e-6 || G > G_prev
                    w = w_old;
                    h = h - 1;
                    iterate = 'no';
                end
                G_prev = G;
                r = AD'*d;
                aux1 = r'*r;
                beta = aux1/aux0;
                p = r + beta*p;
                % Checking if the search direction is a descent direction for the
                % original functional
                DG = AD'*AD*w - AD'*b;
                descent = DG'*p;
                if ( descent>0 )
                    iterate = 'no';
                end
                % New search direction
                aux0 = aux1;
            end

            x = diag(sqrt(theta))*w; % Update with Inexact IAS
            th_old = theta;
            zeta = x./sqrt(theta_0);
            xi = GenGammaUpdate1D(zeta,r_gamma,shape_param);
            theta = theta_0.*xi;
            Dth_a = norm(th_old - theta)/norm(th_old);
            count = count + 1;
            F_a(count) = ObjFcn(x,theta);
            R_a(count) = norm(A*x - b)^2;
            G_a(count) = norm(A*x- b)^2 + sum((x.^2)./theta);
            if count>1
                DF_a = abs(F_a(count)-F_a(count-1))/F_a(count-1);
            end
        end

    elseif (method == "PIAS")
        r = rs(j-1);
        eta = etas(j-1);
        theta_0 = theta_0s(j-1);
        % Prediction Step
        [dxdt,dphidt,y,preconditioner] = Compute_Derivatives(x,phi,r,theta_0,drdt,detadt,dtheta_0dt,A,preconditioner,y_old);

        x = x + dxdt*dt;
        phi = phi + dphidt*dt;
        theta = exp(phi);

        r = rs(j);  %Update hyperparameters
        eta = etas(j);
        theta_0 = theta_0s(j);
        shape_param = (3/2 + eta)/r;
        count = 1;
        while (count <= steps_num)
            M   = [A;spdiags(1./sqrt(theta),0,n,n)];
            rhs = [b;zeros(n,1)];
            shape_param = (3/2 + eta)/r;
            x = M\rhs;   % Update with exact IAS
            zeta = x./sqrt(theta_0);
            xi = GenGammaUpdate1D(zeta,r,shape_param);
            theta = theta_0.*xi;
            phi = log(theta);
            count = count+ 1;
        end
    elseif (method == "Ptheta")
        r = rs(j-1);
        eta = etas(j-1);
        theta_0 = theta_0s(j-1);
        % Prediction Step
        [dxdt,dphidt,y,preconditioner] = Compute_Derivatives(x,phi,r,theta_0,drdt,detadt,dtheta_0dt,A,preconditioner,y_old);
        x = x + dxdt*dt;
        phi = phi + dphidt*dt;
        theta = exp(phi);
        % Update hyperparameters
        r = rs(j);
        eta = etas(j);
        theta_0 = theta_0s(j);
        shape_param = (3/2 + eta)/r;
        count = 1;
        while (count <= steps_num)
            zeta = x./sqrt(theta_0);
            xi = GenGammaUpdate1D(zeta,r,shape_param);
            theta = theta_0.*xi;
            phi = log(theta);
            count = count+ 1;
        end
    end

    X(:,j) = x;
    Theta(:,j) = theta;
    Phi(:,j) = phi;
    j
end



