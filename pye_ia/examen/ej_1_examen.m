
%% a) Resolucion teórica
p=0.01;
syms N
N=solve(0.5-(1-p)^N)
N=-log(2)/log(99/100)

%% a) Grafico de comprobacion teórico
clc, clear, close all
dir='G:\My Drive\AI\CURSO\prob_y_est_ia\examen\';
muestras=300;
p=0.01;
for N=1:muestras
    a(N)=1-binocdf(0,N,p);
end
N=1:muestras;
plot(N,a)
% hold on
% plot(ones(size(N))*68,0:0.7/(length(N)-1):0.7,'--g',ones(size(N))*69,0:0.7/(length(N)-1):0.7,'--r')
% hold on
% plot(N,ones(size(N))*0.5,'c')
xlabel('N');ylabel('1-binocdf(0,N,p)');
grid on
print('-depsc',[dir 'comprob_teo']);
% end
%% b)
N=68;
mu=N*p
sigma=sqrt(N*p*(1-p))
%% c) para encontrar el N


%% c) para media y varianza
% proceso Bernoulli con v.a. uniforme

% configuro la semilla inicial del proceso aleatorio uniforme
rand ('seed', 123);

% probabilidad de fosforo defectuoso
p = 0.01;

% cantidad de fosforos por caja
n=68;

% numero de ensayos
N = 1e5;

% numero de defectos, vale 1 cuando el fosforo es defectuso
ndefectos_vector = zeros(N,1);

% se hacen N ensayos y en cada uno se guardan n fosforos en la caja, se
% buscan qué cantidad de ellos son defectuosos con probabilidad p
for i = 1:N

  % n fosforos por ensayo
  for j = 1:n

    if(rand() < p)
      ndefectos_vector(i) = ndefectos_vector(i) + 1;
    end

  end

end

% calcular la media y varianza
media_muestral = sum(ndefectos_vector) / N

% varianza muestral
varianza_muestral = (1/(N-1)) * sum((ndefectos_vector - media_muestral * ones(N,1)).^2)

% media teorica
media_teorica = n*p

% varianza teorica
varianza_teorica = n * p * (1-p)

