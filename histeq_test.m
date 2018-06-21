% 直方图均衡的实验
close all;
clear, clc;
img = imread('lena.jpg');
img = rgb2gray(im2double(img));
img = img .^ 3;
figure;
subplot(2, 3, 1), imshow(img);
subplot(2, 3, 2 : 3), imhist(img);

g = histeq(img, 256);
subplot(2, 3, 4), imshow(g);
subplot(2, 3, 5 : 6), imhist(g);