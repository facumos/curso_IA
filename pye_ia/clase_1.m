%% Ej. 1
% % a
p=0.5; % probabilidad de ocurrencia
n=10; %cantidad de intentos
k=3; % casos de exito
p_X=factorial(n)/(factorial(k)*factorial(n-k))*p^k*(1-p)^(n-k);

% % b 
p=0.4; % probabilidad de ocurrencia
n=10; %cantidad de intentos
p_X1=0;
for k=1:3; % casos de exito
p_X1=p_X1+factorial(n)/(factorial(k)*factorial(n-k))*p^k*(1-p)^(n-k);
end

% % c
% % Se debe calcular P(B3^B2^B1^A) donde ^ significa intersección
pb=0.6*0.4^3*10;
%%
% % d
% i)
% a)
N=10;
r=rand(1,N);
p=0.5;
H=0;T=0;
for i=1:N
if r(i)<p
    H=H+1;
else
    T=T+1;
end
end 
% i)
% b)
N=10;
r=rand(1,N);
p=0.4;
H=0;T=0;
for i=1:N
if r(i)<p
    T=T+1; % Tail es seca
else
    H=H+1; % Head es cara
end
end
% i)
% c)
%%
% d)
% ii)a)
P=binopdf(3,10,0.5);
% d)
% ii)b)
P1=binocdf(3,10,0.4);
% d)
% ii)c)
P2=0.6*binocdf(3,9,0.4)


%% Ej. 3
a=1-unifcdf(0.7,0,1);
[x,y] = meshgrid(0:0.1:1.5);
mesh(x,y,unifcdf(x,0,1).*unifcdf(y,0,1))
%% Ej. 4
close all
clear
% (1-mvncdf(0.7))*mvncdf(0.4);
[x,y] = meshgrid(0:0.1:4);
p=mvncdf([x y]);
Z = reshape(p,round(length(x)/2-1),round(length(y)/2-1));
surf(x,y,Z)


