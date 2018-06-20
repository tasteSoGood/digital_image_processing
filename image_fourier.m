% 本脚本展示了如何用matlab得到图像的傅里叶频谱图
close all;
clear, clc;
img = imread('lena.jpg');
x = fft2(im2double(rgb2gray(img))); % 傅里叶变换
x = fftshift(x); % 频谱平移
x = real(x); % 只显示实部
x = log(x + 1); % 频谱对数变换
subplot(121);
imshow(rgb2gray(img));
title('original');
subplot(122);
imshow(x, []);
title('frequent');