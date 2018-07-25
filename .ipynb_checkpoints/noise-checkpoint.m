close all;
clear, clc;
img = imread('lena.jpg');
for k = 1 : 1000
    t = imnoise(img, 'gaussian');
    imwrite(t, ['.\\noise\\', num2str(k), '.jpg']);
end