clear;
clc;

% reading steering angle and velocity maneuver
% load('Drv_DeltaSteer.mat');
load('Veh_Vx.mat');
drivingProfile = csvread('parameter.csv',1,0);

% extracting the sampling instances
n = floor(drivingProfile(9)/0.005);
time = floor(size(Drv_DeltaSteer,1)/n);
%delete 'Time.csv'
fileTime = fopen('Time.csv','w+');
fprintf(fileTime,'%10.20f\n',time);
fclose(fileTime);

steeringAngle = zeros(1, time);
velocity = zeros(1, time);
length = size(Drv_DeltaSteer,1);
j = 1;
for i=1:length
    if mod(i,n)==0
        steeringAngle(j) = Drv_DeltaSteer(i);
        velocity(j) = Veh_Vx(i);
        j = j + 1;
    end
end

% parameters
m = drivingProfile(1); %kg
v = max(Veh_Vx); % m/s
lf = drivingProfile(2); %m
lr = drivingProfile(3); %m
l = lr + lf;
Cf = drivingProfile(4); %N/rad
Cr = drivingProfile(5); %N/rad
Iz = drivingProfile(6); %kg m^2

mu = drivingProfile(7);
g = 9.8; %m/s^2
steering_ratio = drivingProfile(8); %15:1
Ku = (m*((lr*Cr)-(lf*Cf)))/(l*Cf*Cr);
max_sideslip = atan(0.02*mu*g);

a11 = (-2)*((Cf+Cr)/(m*v));
a12 = -1-(((2*Cf*lf)-(2*Cr*lr))/(m*v*v));
a21 = (-2)*(((lf*Cf)-(lr*Cr))/Iz);
a22 = (-2)*(((lf*lf*Cf)+(lr*lr*Cr))/(Iz*v));
b1 = (2*Cf)/(m*v);
b2 = (2*Cf*lf)/Iz;
b3 = 0;
b4 = 1/Iz;

% Discretizing the state space
A = [a11 a12;
     a21 a22];
B = [b1 b3;
     b2 b4];
C = [0 1;
    (v*a11) (v*(a12 + 1))];
D = [0 0;
    (v*b1) 0];

sys = ss(A,B,C,D);
Ts = drivingProfile(9);
sys_d = c2d(sys,Ts, 'zoh');

A = sys_d.A;
B = sys_d.B;
C = sys_d.C;
D = sys_d.D;

% Feedback gain computatioon
co = ctrb(sys_d);
isControllable = [rank(A) == rank(co)]

%co = ctrb(sys_d);
%isControllable = [rank(A) == rank(co)]

% Q = 100000000000000*eye(size(A,1));
% Q(1,1) = 10000000000000000000000000000000000000;
% R = eye(size(B,2));
% R(1,1) = 100000000000;
% [K,S,E] = dlqr(A,B,Q,R);

% K=[ 6.7458171835 0.0076120267;
%   0 65536.0];

Q = 200000000000000*eye(size(A,1));
Q(1,1) = 1000000000000000000;
R = eye(size(B,2));
R(1,1) = 10000000000000000;

% dlqr => discrete linear quadratic regulator
[K,S,E] = dlqr(A,B,Q,R);

K = [  1.8639 -0.0279
	  -542794.8896   32604.6225];

abs(eig(A-B*K))

% Observer gain computation
ob = obsv(sys_d);
isObservable = rank(A) == rank(ob);
QN = 1;
RN = eye(size(C));
[KEST,L,P,M,Z] = kalman(sys_d,QN,RN)

L = [ -0.000000000009229 -0.000000000131049
       0.000000000135782   0.000000001407986];

abs(eig(A - L*C))

% Writing K, L values to csv files
delete 'K.csv'
delete 'L.csv'
fileK = fopen('K.csv','w+');
fileL = fopen('L.csv','w+');
fprintf(fileK,'%10.20f %10.20f\n',K.');
fprintf(fileL,'%10.20f %10.20f\n',L.');
fclose(fileK);
fclose(fileL);

% Verifying whether the gain can be used for varying longitudinal velocity

v_min = min(Veh_Vx);
v_max = max(Veh_Vx);

v = v_min;
a11 = (-2)*((Cf+Cr)/(m*v));
a12 = -1-(((2*Cf*lf)-(2*Cr*lr))/(m*v*v));
a21 = (-2)*(((lf*Cf)-(lr*Cr))/Iz);
a22 = (-2)*(((lf*lf*Cf)+(lr*lr*Cr))/(Iz*v));
b1 = (2*Cf)/(m*v);
b2 = (2*Cf*lf)/Iz;
b3 = 0;
b4 = 1/Iz;

A = [a11 a12;
     a21 a22];
B = [b1 b3;
     b2 b4];
C = [0 1;
    (v*a11) (v*(a12 + 1))];
D = [0 0;
    (v*b1) 0];

sys = ss(A,B,C,D);
sys_d = c2d(sys,Ts, 'zoh');

A_min = sys_d.A - (sys_d.B*K);

v = v_max;
a11 = (-2)*((Cf+Cr)/(m*v));
a12 = -1-(((2*Cf*lf)-(2*Cr*lr))/(m*v*v));
a21 = (-2)*(((lf*Cf)-(lr*Cr))/Iz);
a22 = (-2)*(((lf*lf*Cf)+(lr*lr*Cr))/(Iz*v));
b1 = (2*Cf)/(m*v);
b2 = (2*Cf*lf)/Iz;
b3 = 0;
b4 = 1/Iz;

A = [a11 a12;
     a21 a22];
B = [b1 b3;
     b2 b4];
C = [0 1;
    (v*a11) (v*(a12 + 1))];
D = [0 0;
    (v*b1) 0];

sys = ss(A,B,C,D);
sys_d = c2d(sys,Ts, 'zoh');

A_max = sys_d.A - (sys_d.B*K);

P = [2 1;1 1];
isAminCtrl = det(A_min'*P*A_min-P)<0;
isAmaxCtrl = det(A_max'*P*A_max-P)<0;

% Evaluating system
plot_sideslip = zeros(1, time);
plot_sidesliphat = zeros(1, time);
plot_yaw = zeros(1, time);
plot_yawhat = zeros(1, time);
plot_u = zeros(1, time);

x = [0;0];
xhat = [0;0];
y = C*x;
yhat = C*xhat;
u = [0;0];

for i=1:time
    delta = steeringAngle(i)*(pi/180)*steering_ratio;    
    v = velocity(i);
    
    a11 = (-2)*((Cf+Cr)/(m*v));
    a12 = -1-(((2*Cf*lf)-(2*Cr*lr))/(m*v*v));
    a21 = (-2)*(((lf*Cf)-(lr*Cr))/Iz);
    a22 = (-2)*(((lf*lf*Cf)+(lr*lr*Cr))/(Iz*v));
    b1 = (2*Cf)/(m*v);
    b2 = (2*Cf*lf)/Iz;
    b3 = 0;
    b4 = 1/Iz;

    A = [a11 a12;
         a21 a22];
    B = [b1 b3;
         b2 b4];
    C = [0 1;
        (v*a11) (v*(a12 + 1))];
    D = [0 0;
        (v*b1) 0];
    
    sys = ss(A,B,C,D);
    Ts = 0.04;
    sys_d = c2d(sys,Ts, 'zoh');
    
    A = sys_d.A;
    B = sys_d.B;
    C = sys_d.C;
    D = sys_d.D;
    
    a = A(1,2)/A(2,2);
    b = B(1,1) - ((B(2,1)*A(1,2))/A(2,2));
    c = v/(A(1,1) - ((A(2,1)*A(1,2))/A(2,2)));
    
    
    r = y - yhat;
    x = A*x + B*u;
    xhat = A*xhat + B*u + L*r;
    u = -(K*xhat) + [delta;0];
    y = C*x + D*u;
    yhat = C*xhat + D*u;
    
    plot_sideslip(i) = x(1);
    plot_sidesliphat(i) = xhat(1);
    plot_yaw(i) = x(2);
    plot_yawhat(i) = xhat(2);
    plot_u(i) = u;
end

fontsize = 10;
linewidth = 1;

clf;
subplot(2,2,1);
hold on;
plot(plot_yaw,'Linewidth',linewidth);
plot(plot_yawhat);
set(gca,'FontSize',fontsize)
xlabel('Time(x40x10^{-3})(s)','FontSize',fontsize);
ylabel('rad/s','Fontsize',fontsize);
legend({'yaw rate','estimated yaw rate'},'FontSize',fontsize);
% axis([0 time -0.2 0.2])
grid on;
hold off;

subplot(2,2,2);
hold on;
plot(plot_sideslip,'Linewidth',linewidth);
plot(plot_sidesliphat);
set(gca,'FontSize',fontsize)
xlabel('Time(x40x10^{-3})(s)','FontSize',fontsize);
ylabel('rad','Fontsize',fontsize);
legend({'side slip','estimated side slip'},'FontSize',fontsize);
% axis([0 time -0.2 0.2])
grid on;
hold off;

subplot(2,2,3);
hold on;
plot(plot_u(1),'Linewidth',linewidth);
plot(plot_u(2));
set(gca,'FontSize',fontsize)
xlabel('Time(x40x10^{-3})(s)','FontSize',fontsize);
ylabel('rad','Fontsize',fontsize);
legend({'steering angle','Mz'},'FontSize',fontsize);
% axis([0 time -0.2 0.2])
grid on;
hold off;
