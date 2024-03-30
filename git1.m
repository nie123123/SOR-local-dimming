% Read the image
img = imread('your picture');
img = double(img);

% Read the input image XH
XH = img;
% Downsample
XL = imresize(XH, [48, 64], 'bicubic'); % Downsample the image using bicubic interpolation

% Get the components for each color channel
redChannel = double(XL(:,:,1));
greenChannel = double(XL(:,:,2));
blueChannel = double(XL(:,:,3));

% Apply DCT transformation to each color channel
dctRed = dct2(redChannel);
dctGreen = dct2(greenChannel);
dctBlue = dct2(blueChannel);
% Calculate image signature
IS_R = sign(dctRed); % Discrete Cosine Transform (DCT) on the downsampled image and take the sign
IS_G = sign(dctGreen);
IS_B = sign(dctBlue);

% Reconstruct the image using inverse transformation
reconstructed_XLr = idct2(IS_R); % Inverse Discrete Cosine Transform (IDCT) on the signature image to reconstruct the image
reconstructed_XLg = idct2(IS_G);
reconstructed_XLb = idct2(IS_B);

% Calculate initial saliency map
SM_R = conv2(reconstructed_XLr .* reconstructed_XLr, fspecial('gaussian', [5 5], 1), 'same'); % Compute initial saliency map using Gaussian smoothing filter
SM_G = conv2(reconstructed_XLg .* reconstructed_XLg, fspecial('gaussian', [5 5], 1), 'same');
SM_B = conv2(reconstructed_XLb .* reconstructed_XLb, fspecial('gaussian', [5 5], 1), 'same');
SM_H1= 0.2989*SM_R + 0.5870 *SM_G+0.1140 *SM_B;
% Upsample to get high-resolution saliency map
SM_H = imresize(SM_H1,  [1080, 1920], 'bicubic'); % Upsample the initial saliency map to the original image size using bicubic interpolation
% Find the maximum and minimum values in the matrix
maxValue = max(SM_H(:));
minValue = min(SM_H(:));
% Normalize the matrix
SM_H = (SM_H - minValue) / (maxValue - minValue);
S=SM_H;

% Calculate image parameters
[row, col, ~] = size(img);
N = row * col;
lambda_row = row;
lambda_col = col;
lambda_color = 3;
gamma = 2.2;
S_max = max(S(:));

% Calculate delta 
delta= mean(S(:)) + std(S(:));
% Truncate W_es
S_T = zeros(size(S));
S_T(S >= delta) = S(S >= delta) / double(S_max);

% Calculate IMAX and target MSE
max_R = max(img(:, :, 1), [], 'all');
max_G = max(img(:, :, 2), [], 'all');
max_B = max(img(:, :, 3), [], 'all');
IMAX = max([max_R, max_G, max_B]);
target_PSNR = 35; % Target PSNR
target_MSE = (255^2) / (10^(target_PSNR/10));

% Truncated RGB image
truncatedImage = img;
truncatedImage(repmat(~S_T, [1, 1, size(img, 3)])) = 0;

% Calculate target total variance
num_pixels = sum(S_T(:) > 0);
lambda =num_pixels*lambda_color;
target_TSE = target_MSE * lambda;

% Initialize maximum cropping point and minimum total variance
ICCP=IMAX-1;
TSEC_MAX = Inf;

% Find the optimal cropping point
for i = IMAX:-1:0
    TSE_R = sum(sum((truncatedImage(:, :, 1) > i) .* (truncatedImage(:, :, 1) - i).^2));
    TSE_G = sum(sum((truncatedImage(:, :, 2) > i) .* (truncatedImage(:, :, 2) - i).^2));
    TSE_B = sum(sum((truncatedImage(:, :, 3) > i) .* (truncatedImage(:, :, 3) - i).^2));
    TSEC_MAX_new = TSE_R + TSE_G + TSE_B;
    
    if TSEC_MAX_new < target_TSE
        ICCP = i;
        TSEC_MAX = TSEC_MAX_new;
    end
end

Z=ones(1080,1920)*ICCP;

% Original displayed image
BL_0=1;

% Dimming based on cropping point

% Dimming value reduction
BL = BL_0.*(ICCP/255);
% Pixel compensation
IMG = img.*(255/ICCP);
IMG(IMG>255)=255;
% Display image after dimming
IMAGE = IMG.*BL;

% Calculate PSNR
mse = immse(image, IMAGE);
psnr = 10 * log10((255^2) / mse);

% Calculate SSIM
ssim_value = ssim(image, IMAGE);

% Calculate PPR
ppr=(255-ICCP)/255*100;