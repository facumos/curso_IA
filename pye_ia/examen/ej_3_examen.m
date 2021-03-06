%% c) Intervalo de confianza
randn('seed', 1234)
% numero de simulaciones
n = 2e4;
% numero de muestras
N = 10;
% media muestral
% la usamos como media de distribucion en el ejercicio
mu = 48;
% dispersion
sigma = 4;
% el intervalo de confianza de 95
% teorico es:
mu_min_teorico = mu - 1.96*sigma/sqrt(N);
mu_max_teorico = mu + 1.96*sigma/sqrt(N);
% nivel de confianza del intervalo, deberia ser igual a 0.95 al final de 
% la simulacion
confianza = 0;
for i=1:n    
    % generar distribucion    
    % esta surge de la estimacion de media muestral    
    X = sigma/sqrt(N)*randn(N,1) + mu;    
    %  chequeo cuantos valores estan en el intervalo de confianza
    confianza = confianza + (1/(n*N)) *sum(X>=mu_min_teorico & X<=mu_max_teorico);
end
% estimacion del intervalo de confianza
confianza_simulado = confianza 
confianza_teorico = 0.95
