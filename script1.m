% Specify damping coefficient.
c = 5;   
% Specify stiffness.
k = 300; 
% Specify load command.
u = 1:10;
% Specify mass.
m = 10*u + 0.1*u.^2;
% Compute linear system at a given mass value.
for i = 1:length(u)
   A = [0 1; -k/m(i), -c/m(i)];
   B = [0; 1/m(i)];
   C = [1 0];
   sys(:,:,i) = ss(A,B,C,0); 
end