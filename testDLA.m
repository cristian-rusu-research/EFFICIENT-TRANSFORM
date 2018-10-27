%% clear everything
close all
clear
clc

%% read input data
% images = {'images/lena.bmp', 'images/peppers.bmp', 'images/boat.bmp'};
images = {'images/lena.bmp'};
cI = readImages(images);
cI = cI./255;

%% parameters
% the sparsity
k0 = 4;
% the number of reflectors
h = 3;
% the number of G and R transforms
m = 85;

%% call the algorithms, takes several minutes
[Uh, Xh, theUsh, tush, errh] = h_dla(cI, k0, h);
[Ug, Xg, positionsg, valuesg, tusg, errg] = g_dla(cI, k0, m);
[Ur, Xr, Dr, positionsr, valuesr, tusr, errr] = r_dla(cI, k0, m, []);
