A=[1 2;2 1];
[P,D]=eig(A);
P2=[2 -2;3 3];
A2=inv(P2)*D*P2
%%
A=[1/3 -1/3;4/3 5/3];
% [P,D]=eig(A) % Los autovectores son LD, porque los autovalores tienen
% multplicidad 2
[P,J]=jordan(A)