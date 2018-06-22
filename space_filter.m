% ÏßÐÔ¿Õ¼äÂË²¨Æ÷
close all;
clear, clc;
img = imread('lena.jpg');
img = im2double(rgb2gray(img));
figure, imshow(img);
w = ones(5);
gd = imfilter(img, w, 'conv', 'replicate', 'same');
figure, imshow(gd, [])