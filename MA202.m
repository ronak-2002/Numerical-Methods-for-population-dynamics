clear;
clc;

%Parameter Estimation:

x= [53; 57.4; 62; 69.8; 76; 84.2; 89.9; 97.9];                             % Population values
y = [0.733333333; 0.92; 1.56; 1.24; 1.64; 1.14; 1.6; 1.48];                % Values for delta_P/delta_t
p1 = polyfit(x, y, 1);                                                     % Applying linear least square fit
f = polyval(p1,x);                                                           
figure(1)   
nexttile
plot(x,y,'o',x,f,'-')                                                      % Plotting the data with its linear fit 
title('Plot between rate of change of population and population')
legend('Population Data','Linear fit')
xlabel('P')
ylabel('delta P/delta t')

z = [0.013836478; 0.016027875; 0.02516129; 0.017765043; 0.021578947; 0.013539192; 0.017797553; 0.015117467];   % Values for delta_P/(delta_t * P)
p2 = polyfit(x, z, 1);                                                       
f = polyval(p2,x);                                                         % Applying linear least square fit
nexttile
plot(x,z,'o',x,f,'-')                                                      % Plotting the data with its linear fit
title('Plot between rate of change of population/Population and population')
legend('Population Data','Linear fit')
xlabel('P')
ylabel('delta P/(delta t * P')

%Specifying a time array for the collected data
X = (1966:1:2007);
t = [1966; 1972; 1977; 1982; 1987; 1992; 1997; 2002; 2007];
% Lagrange interpolation for given data
x = [53; 57.4; 62; 69.8; 76; 84.2; 89.9; 97.9; 105.3];
Y = zeros(42,1);
k = 1965;
for i = 1:42
    k = k+1;
    Y(i) = Lagrange(t, x, 8, k);
end
disp('Interpolations made from Actual data:')

for i = 1:length(X)
    C = ['Population at time = ', num2str(X(i)), ' is ', num2str(Y(i))];
    disp(C)
end

% Linear fits in form of an equation y = ax + b are obtained, where p1 and p2 contain the values of a & b

% Exponential Model where P' = kP
k = p1(1);
f_exp = @(p) (k*p);                                                        % P' for exponential model
y1 = Runge_Kutta(f_exp, 53, 1, 41);
y2 = Euler(f_exp, 53, 1, 41);
disp(newline)
disp('Solutions for Exponential model by 4th-order Runge-Kutta method:')
for i = 1:length(X)
    C = ['Population at time = ', num2str(X(i)), ' is ', num2str(y1(i))];
    disp(C)
end
disp(newline)
disp("Solutions for Exponential model by Euler's method:")
for i = 1:length(X)
    C = ['Population at time = ', num2str(X(i)), ' is ', num2str(y2(i))];
    disp(C)
end

% Hyperbolic Model where P' = kP^2/P0 = cP^2
c = p2(1);
f_hyp = @(p) (c*p*p);                                                      % P' for hyperbolic model
y3 = Runge_Kutta(f_hyp, 53, 1, 41);
y4 = Euler(f_hyp, 53, 1, 41);
disp(newline)
disp("Solutions for Hyperbolic model by 4th-order Runge-Kutta method:")
for i = 1:length(X)
    C = ['Population at time = ', num2str(X(i)), ' is ', num2str(y3(i))];
    disp(C)
end
disp(newline)
disp("Solutions for Hyperbolic model by Euler's method:")
for i = 1:length(X)
    C = ['Population at time = ', num2str(X(i)), ' is ', num2str(y4(i))];
    disp(C)
end

% Logistic Model where P' = cLP(1- P/L)
c = p2(1) * (-1);
L = p2(2)/c;
f_log = @(p) (c*L*p*(1 - p/L));                                            % P' for logistic model
y5 = Runge_Kutta(f_log, 53, 1, 41);
y6 = Euler(f_log, 53, 1, 41);
disp(newline)
disp("Solutions for Logistic model by 4th-order Runge-Kutta method:")
for i = 1:length(X)
    C = ['Population at time = ', num2str(X(i)), ' is ', num2str(y5(i))];
    disp(C)
end
disp(newline)
disp("Solutions for Logistic model by Euler's method:")
for i = 1:length(X)
    C = ['Population at time = ', num2str(X(i)), ' is ', num2str(y6(i))];
    disp(C)
end

% Errors for Exponential model
error1 = error(y1, Y);
error2 = error(y2, Y);

% Errors for Exponential model
error3 = error(y3, Y);
error4 = error(y4, Y);

% Errors for Exponential model
error5 = error(y5, Y);
error6 = error(y6, Y);
disp(newline)
disp("% Errors in the different methods:")
C = [ newline, 'Error in 4th-order Runge-Kutta method for exponential model is ', num2str(error1)];
disp(C)
C = [ newline, 'Error in Euler method for exponential model is ', num2str(error2)];
disp(C)
C = [ newline, 'Error in 4th-order Runge-Kutta method for Hyperbolic model is ',num2str(error3)];
disp(C)
C = [ newline, 'Error in Euler method for Hyperbolic model is ', num2str(error4)];
disp(C)
C = [ newline, 'Error in 4th-order Runge-Kutta method for Logistic model is ', num2str(error5)];
disp(C)
C = [ newline, 'Error in Euler method for Logistic model is ', num2str(error6)];
disp(C)

% Plotting the different data
figure(2)
plot(X, Y, '-', X, y1, '-', X, y2, '-', 'linewidth', 1)                                                      
title('Exponential Model')
legend('Actual data','Runge-Kutta Method', 'Euler Method')
xlabel('t')
ylabel('P')

figure(3)
plot(X, Y, '-', X, y3, '-', X, y4, '-', 'linewidth', 1)                                                      
title('Hyperbolic Model')
legend('Actual data','Runge-Kutta Method', 'Euler Method')
xlabel('t')
ylabel('P')

figure(4)
plot(X, Y, '-', X, y5, '-', X, y6, '-', 'linewidth', 1)                                                      
title('Logistic Model')
legend('Actual data','Runge-Kutta Method', 'Euler Method')
xlabel('t')
ylabel('P')

figure(5)
plot(X, Y, '-', X, y1, '-', X, y3, '-', X, y6, '-', 'linewidth', 1)                                                      
title('Comparison of different models')
legend('Actual data','Exponential Model', 'Hyperbolic Model', 'Logistic Model')
xlabel('t')
ylabel('P')

function y = Runge_Kutta(f, p0, h, n)

     y = zeros(n+1,1);
     y(1) = p0;
     for i = 2:n+1
        k1 = f(p0);
        k2 = f(p0 + k1*h/2);
        k3 = f(p0 + k2*h/2);
        k4 = f(p0 + k3*h);

        p1 = p0 + (k1 + 2*k2 + 2*k3 + k4)*h/6;
        y(i) = p1;
        p0 = p1;
    end
end

function y = Euler(f, p0, h, n)

     y = zeros(n+1,1);
     y(1) = p0;
     for i = 2:n+1
         
        p1 = p0 + h*f(p0);
        y(i) = p1;
        p0 = p1;
    end
end

function sum = Lagrange(x, y, n, X)

    sum = 0;
    for i = 1: n+1
        prod = y(i);
        for j = 1: n+1
            if (i ~= j)
                prod = prod*((X-x(j))/(x(i)-x(j)));
            end
        end
        sum = sum + prod;
    end
end

function e = error(y1, y2)
    
    ea = zeros(length(y1),1);
    for i = 1:length(y1)
        ea(i) = abs(y1(i) - y2(i))/y2(i)*100;
    end
    e = sum(ea)/length(y1);
end
        


