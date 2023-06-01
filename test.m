clear;close;clc;

N = 12;
load('test_data.mat')
data1 = input(1:30,:);
data2 = L_slot(1:30,:);

pred = matpy(data1, data2);
pred = double(pred);

figure;
hold on;
plot(angle(output(N,:)));
plot(angle(pred(N,:)));

title("测试集验证结果");
legend("FDTD仿真","NN预测");


figure;
hold on;
plot(abs(output(N,:)));
plot(abs(pred(N,:)));

title("测试集验证结果");
legend("FDTD仿真","NN预测");

