clc,clear, close all
%% Datos
pos=dlmread('G:\My Drive\AI\CURSO\matematica_ia\practica\Ejercicios\Simulaciones del Filtro de Kalman\posicion.dat');
vel=dlmread('G:\My Drive\AI\CURSO\matematica_ia\practica\Ejercicios\Simulaciones del Filtro de Kalman\velocidad.dat');
pos=pos(:,2:end);
vel=vel(:,2:end);
%% Condiciones Iniciales
x_0I0=[10.7533, 36.6777, -45.1769, 1.1009, -17.0, 35.7418, -5.7247, 3.4268, 5.2774]';
P_0I0=diag([100, 100, 100, 1, 1, 1, 0.01, 0.01, 0.01],0);
Q=0.3*eye(9);
%% Modelo
h=1;
dim=3; %asumo, por las condiciones iniciales que cada estado tiene 3 dimensiones
A=[eye(dim) h.*eye(dim) h^2/2.*eye(dim);zeros(dim) eye(dim) h.*eye(dim);zeros(dim) zeros(dim) eye(dim)];
B=[eye(dim);eye(dim);eye(dim)];
C=[eye(dim) zeros(dim) zeros(dim)];
%% 1
t=1:length(pos);
sigma=10;
R=sigma^2*eye(dim);
ruido=normrnd(0,sigma,size(pos));
y_n=pos'+ruido';

x_acu=kalman_f(A,B,C,x_0I0,P_0I0,Q,R,y_n,t,dim);
y_est=C*x_acu;
plot(t,pos,t,y_est)
error=pos'-y_est;
figure;plot(t,error); xlim([100,t(end)])
MSE_1=sum(error.^2,2)/length(t);
% err_1 = immse(pos',y_est);
%% 2
t=1:length(pos);
sigma=10;
R=sigma^2*eye(dim);
ruido= [sigma*(2*rand(length(t),1)-1) sigma*(2*rand(length(t),1)-1) sigma*(2*rand(length(t),1)-1)];
y_n=pos'+ruido';

x_acu=kalman_f(A,B,C,x_0I0,P_0I0,Q,R,y_n,t,dim);
y_est=C*x_acu;

figure; plot(t,pos,t,y_est)
error=pos'-y_est;
figure; plot(t,error); xlim([100,t(end)])
MSE_2=sum(error.^2,2)/length(t);
% err_2 = immse(pos',y_est);

%% 3
t=1:length(pos);
sigma_p=10;
sigma_v=0.2;
R=[sigma_p^2*eye(dim) zeros(dim); zeros(dim) sigma_v^2*eye(dim)];
ruido_p=normrnd(0,sigma_p,size(pos));
ruido_v=normrnd(0,sigma_v,size(vel));
y_n=[pos'+ruido_p'; vel'+ruido_v'];
C=[eye(dim) zeros(dim) zeros(dim);zeros(dim) eye(dim) zeros(dim)];

x_acu=kalman_f (A,B,C,x_0I0,P_0I0,Q,R,y_n,t,dim);
y_est=C*x_acu;
figure; plot(t,pos,t,y_est(1:3,:))
figure; plot(t,pos,t,y_est(4:6,:))

error_p=pos'- y_est(1:3,:);
MSE_p3=sum(error_p.^2,2)/length(t);
figure; plot(t,error_p);xlim([100,t(end)])

error_v=vel'- y_est(4:6,:);
MSE_v3=sum(error_v.^2,2)/length(t);



