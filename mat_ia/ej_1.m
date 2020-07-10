A=[0 1;-1 -1];
B=[0;1];
C=[1 0];
sys=ss(A,B,C,0);
[num,den]=ss2tf(A,B,C,0);
G=tf(num,den);
s=tf('s')
P=G*1/(s+1)
[n,d]=tfdata(P,'v');
[r,p,k]=residue(n,d)
%%
syms s t
U=laplace(exp((-t)))
ilaplace(1/(s^3 + 2*s^2 + 2*s + 1));
t=0:0.01:10;
p_t=exp(-t) - exp(-t/2).*(cos((3^(1/2)*t)/2) - (3^(1/2).*sin((3^(1/2).*t)./2))./3);
% plot(t,p_t,tout,p)

%%
syms s t
U=laplace(exp(-(t-1)));
ilaplace(U/(s^2+s+1));
t=0:0.01:10;
p_t=exp(-t).*exp(1) - exp(-t/2).*exp(1).*(cos((3^(1/2).*t)./2) - (3^(1/2).*sin((3^(1/2).*t)./2))./3);
plot(t,p_t,tout,p)
%%
syms s t
U=laplace(exp(1i*2*pi*t))
ilaplace(U/(s^2+s+1));
t=0:0.01:10;
p_t=exp(pi.*t.*2i)./(pi*2i - 4*pi^2 + 1) - (exp(-t/2).*(cos((3^(1/2).*t)./2) + (2*3^(1/2).*sin((3^(1/2).*t)./2).*(1/2 + pi*2i))/3))/(pi*2i - 4*pi^2 + 1);
plot(t,real(p_t),tout,p)
figure;plot(t,imag(p_t),tout,p1)