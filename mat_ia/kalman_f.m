function x_acu=kalman_f (A,B,C,x_0I0,P_0I0,Q,R,y_n,t,dim)
x_acu=zeros(9,length(t));
x_n_1In_1=x_0I0;
P_n_1In_1=P_0I0;
for i=1:t(end)
    x_nIn_1=A*x_n_1In_1;
%   P_nIn_1=A*P_n_1In_1*A'+B'*Q*B;  % Hay un problema con las dimensiones acá
    P_nIn_1=A*P_n_1In_1*A'+Q;
%   K_n=P_nIn_1*C'\(C*P_nIn_1(i)*C'+R);  % Acá hay otro problema con las dimensiones
%     K_n=C*P_nIn_1\(C*P_nIn_1*C'+R);
    K_n=P_nIn_1*C'*inv(C*P_nIn_1*C'+R);
    x_nIn=x_nIn_1+K_n*(y_n(:,i)-C*x_nIn_1);
    P_nIn=(eye(dim^2)-K_n*C)*P_nIn_1;
    P_n_1In_1=P_nIn;
    x_acu(:,i)=x_nIn;
end
end