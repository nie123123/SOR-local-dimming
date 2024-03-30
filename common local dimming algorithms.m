img = imread('input picture');     % Input image
r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);   
z = zeros(1080,1920);
for i = 1:1080
    for j = 1:1920
        z(i,j) = max(max(r(i,j),g(i,j)),b(i,j));
    end
end     % Sub-pixel maximum method to extract luminance matrix
A = mat2cell(z,[repelem(40, 27)],[repmat(40, 1, 48)]);  % Partition operation
D = cellfun(@max,cellfun(@max,A,'UniformOutput',false)); % Maximum method to extract backlight
D = cellfun(@mean,cellfun(@mean,A,'UniformOutput',false)); % Mean method to extract backlight
C = cellfun(@mean,cellfun(@mean,A,'UniformOutput',false));
D = (C/255).^(1/2).*255;    % Root mean square method to extract backlight
Z = zeros(1080,1920);
row1 = 0:40:1080; 
row2 = row1;
row1 = row1 + 1;
col1 = 0:40:1920; 
col2 = col1;
col1 = col1 + 1;
for m = 1:27
    for n = 1:48
        Z(row1(m):row2(m+1), col1(n):col2(n+1)) = D(m, n);   % Determine the entire backlight distribution map as a 27×48 matrix
    end
end
w = [0.06 0.11 0.06;0.08 0.38 0.08;0.06 0.11 0.06];
E = imfilter(D, w,'conv','replicate','same');
L = imresize(E,2,'bilinear');
F = imfilter(L, w,'conv','replicate','same');
G = imresize(F,2,'bilinear');
H = imfilter(G, w,'conv','replicate','same');
I = imresize(H,[1080,1920],'bilinear');        % BMA smoothing, triple convolution diffusion, the last one directly expands to 1080×1920 resolution, using bilinear interpolation
a = 255 * ones(1080,1920);
b = a ./ I;
R = double(img(:,:,1));
G = double(img(:,:,2));
B = double(img(:,:,3));
II(:,:,1) = R .* b;
II(:,:,2) = G .* b;
II(:,:,3) = B .* b;
II(II>255) = 255;    % Linear pixel compensation 
IMG(:,:,1) = II(:,:,1) .* I / 255;
IMG(:,:,2) = II(:,:,2) .* I / 255;
IMG(:,:,3) = II(:,:,3) .* I / 255;    % Final dimming image
img = double(img);
RR = II(:,:,1);
GG = II(:,:,2);
BB = II(:,:,3);
RR(RR<=255) = 0;
GG(GG<=255) = 0;
BB(BB<=255) = 0;
RGB = RR + GG + BB;
OF = sum(RGB(:) > 0);
OFR = (OF / (1920 * 1080)) * 100;   % Pixel overflow rate calculation

mse1 = immse(img, IMG);
psnr = 10 * log10((255^2) / mse1);   % Calculate PSNR
 
ssim_value = ssim(img, IMG);  % Calculate SSIM

p0 = sum(sum(a));
p1 = sum(sum(Z));
PPR = (p0 - p1) / p0 * 100;   % Calculate PPR