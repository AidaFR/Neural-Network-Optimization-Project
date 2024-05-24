% Citirea datelor din fisierul CSV
data = readmatrix('mushroom_cleaned_Amestecat.csv');

% Calculeaza numarul de randuri pentru setul de antrenare și testare
numar_antrenare = round(0.8 * size(data, 1));

% Separa datele în seturi de antrenare si testare
date_antrenare = data(1:numar_antrenare, :);
date_testare = data(numar_antrenare+1:end, :);

% Afiseaza dimensiunile seturilor de antrenare și testare
disp(['Dimensiunea setului de antrenare: ', num2str(size(date_antrenare, 1))]);
disp(['Dimensiunea setului de testare: ', num2str(size(date_testare, 1))]);

% Extrage ultima coloana din setul de antrenare ai creeaza vectorul de etichete pentru antrenare
etichete_antrenare = date_antrenare(:, end);
% Asemenea pentru testare
etichete_testare = date_testare(:, end);

m=16; % numarul de neuroni
N=800;%numarul de exemple
n=9;%numarul de caracteristici ale unui exemplu
x = randn(m, 1); % Vectorul de ponderi pentru stratul ascuns
X = randn(n+1, m); % Matricea de ponderi pentru stratul de intrare
A_concatenated = [date_antrenare ones(N, 1)];%matricea de intrare careia ii adaugam o coloana suplimentara cu 1 pentru bias
alpha0_x=0.1; %pasul initial pentru derivata partiala in raport cu x/ learning rate
alpha0_X=0.1; %pasul initial pentru derivata partiala in raport cu X
vect=[]; %vector de convergenta
norme_gradient = [];
crit=1;
epsi=0.001;%precizia pentru criteriul de oprire
% Măsurare timp de început
tic;
%Clculul iesirii initiale
y = sigmoid_with_shift_scale(A_concatenated*X)*x;
f_previous = Loss(etichete_antrenare, y);
 disp(f_previous)
%% Algoritmul gradientului descendent
t=0;
while crit > epsi
      % Calculul gradientilor pentru parametrii x si X
    grad_loss_x = zeros(m, 1);
    grad_loss_X = zeros(n+1, m); 
%[grad_loss_x, grad_loss_X] = Calculate_gradients_loss_xX(etichete_antrenare, y, X, x, A_concatenated);
    sum=0;
    for i = 1:N
       sum=sum+(etichete_antrenare(i)/ y(i) - ((1 - etichete_antrenare(i)) / (1 - y(i))));
       sum=sum*(-1/N);

        a_i = A_concatenated(i, :);
        grad_loss_x = (sigmoid_with_shift_scale(a_i*X))'*sum;
        aux=sigmoid_deriv(a_i*X);
        grad_loss_X = sum*a_i'*x'.*aux;
    end
 
        % Calculul pasului si normei gradientului
        alpha_x=alpha0_x/ (norm(grad_loss_x))^2;
        alpha_X=alpha0_X/ (norm(grad_loss_X))^2;
    
        % Actualizarea parametrilor folosind gradientul descendent
        x = x - alpha_x * grad_loss_x;
        X = X - alpha_X * grad_loss_X;
    
        % Verificarea criteriului de oprire       
        f_current = Loss(etichete_antrenare,sigmoid_with_shift_scale(A_concatenated*X)*x );
        crit=abs(f_current - f_previous);
        %disp(f_current)
        %disp(crit)
        % Actualizarea valorii anterioare a fct obiectiv
        f_previous=f_current;
        alpha0_x=alpha_x;
        alpha0_X=alpha_X;
        % Măsurarea normei gradientului si retinerea criteriului
        grad_norm = norm([grad_loss_x(:); grad_loss_X(:)]);
        vect=[vect,crit];
         % Afisarea progresului
    disp(['Iteratie: ', num2str(t), ', Norma gradientului: ', num2str(grad_norm)]);
    
    % Salvarea valorii normei gradientului pentru această iterație
    norme_gradient = [norme_gradient, grad_norm];
        % Incrementarea numarului de iteratii
        t=t+1;
         
end  
%Sa
    x_opt = x;
    X_opt = X;
%disp(x_opt)
% Măsurarea timpului de sfârșit și afișarea timpului total de execuție
timp_executie = toc;
disp(['Timp de executie: ', num2str(timp_executie), ' secunde']);
% Graficul evoluției normei gradientului si a functiei obiectiv
figure;
subplot(2,1,1);
plot(1:t, norme_gradient, 'b');
title('Evolutia normei gradientului descendent');
xlabel('Iterație');
ylabel('Norma gradientului');
grid on
subplot(2,1,2);
plot(1:t, vect, 'b');
title('Evol functiei obiectiv');

% Calcularea etichetelor prezise pe setul de testare
y_pred =sigmoid_with_shift_scale(A_concatenated*X_opt)*x_opt;
y_pred = y_pred(1:size(etichete_testare, 1));
% Calcularea matricei de confuzie
C = confusionmat(etichete_testare, y_pred);

% Afisarea matricei de confuzie
disp('Matricea de confuzie:');
disp(C);

% Calcularea metricilor de performanta
TP = C(1, 1);
FP = C(1, 2);
FN = C(2, 1);
TN = C(2, 2);

precizie = (TP + TN) / sum(C(:));
sensibilitate = TP / (TP + FN);
specificitate = TN / (TN + FP);

disp(['Precizie: ', num2str(precizie)]);
disp(['Sensibilitate: ', num2str(sensibilitate)]);
disp(['Specificitate: ', num2str(specificitate)]);

%% Algoritmul gradientului stocastic
while crit > epsilon
    % Amestecarea setului de antrenare la fiecare epoca
    A_concatenated = A_concatenated(randperm(N), :);
    
    % Actualizarea ponderilor pentru fiecare exemplu de antrenare
    for i = 1:N
        a_i = A_concatenated(i, :);
        
        % Calculul gradientilor pentru parametrii x și X folosind exemplul curent
       sum= (etichete_antrenare(i) / y(i) - ((1 - etichete_antrenare(i)) / (1 - y(i))));
        sum=sum * (-1);
        grad_loss_x = sum *( sigmoid_with_shift_scale(a_i*X))';
        aux = sigmoid_deriv( a_i*X);
        grad_loss_X = sum * a_i'* x'.* aux;
        
        % Calculul pasului folosind norma grad
        alpha_x = alpha0_x / norm(grad_loss_x);
        alpha_X = alpha0_X / norm(grad_loss_X);
        
        % Actualizarea parametrilor
        x = x - alpha_x * grad_loss_x;
        X = X - alpha_X * grad_loss_X;
        
        % Actualizarea iesirii și fct obiectiv
        y = sigmoid_with_shift_scale(A_concatenated * X) * x;
        f_current = Loss(etichete_antrenare(i), y);
        
        crit = abs(f_current - f_previous);
        f_previous = f_current;  
        % Incrementarea nr total de iteratii
        t=t+1;
        
        % Afisarea progresului 
        if mod(t, 100) == 0
            disp(['Iteratie: ', num2str(t), ', Norma gradientului: ', num2str(norm([grad_loss_x(:); grad_loss_X(:)])), ', Criteriul de oprire: ', num2str(crit)]);
        end
    end
end



% Plotarea evoluției funcției obiectiv și a normei gradientului
figure;
subplot(2,1,1);
plot(1:t, vect, 'b');
title('Evoluția funcției obiectiv');
xlabel('Iterații');
ylabel('Valoare funcție obiectiv');
grid on;

subplot(2,1,2);
semilogy(1:t, norm(grad_loss_x) + norm(grad_loss_X), 'r');
title('Evoluția normei gradientului stocastic');
xlabel('Iterații');
ylabel('Norma gradientului');
grid on;
