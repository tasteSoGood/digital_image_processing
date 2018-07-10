% —È÷§fft
clc, clear;
close all;

a = [1 3 4 5];
b = [6 0 1 2 5];

disp(conv(a, b));

% ¿©≥‰
a(9) = 0;
b(9) = 0;
disp(ifft(fft(a) .* fft(b)));