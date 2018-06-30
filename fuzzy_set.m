% 基于模糊集合的图像增强
close all;
clear, clc;
img = imread('lena.jpg');
img = rgb2gray(img);
subplot(121), imshow(img);
% 参数
Fd = 0.8, FD = -Fd, Fe = 128, Xmax = 255;
threshold = 0.5;
x = double(img);
% 模糊特征平面
P = (1 + (Xmax - x) ./ Fe) .^ FD;
% 模糊增强
times = 1;
for k = 1 : times
    t = P;
    t(P <= threshold) = 2 .* P(P <= threshold) .^ 2;
    t(P > threshold) = 1 - 2 .* (1 - P(P > threshold)) .^ 2;
    P = t;
end
% 反模糊化
I = Xmax - Fe * ((1 ./ P) .^ (1 / Fd) - 1);
subplot(122), imshow(uint8(I));