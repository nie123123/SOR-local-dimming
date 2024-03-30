% Read the image
img = imread('input picture');
img_sal = imread('salient ranking map');     % Read the saliency map

% Extract the RGB channels
r = img(:, :, 1);
g = img(:, :, 2);
b = img(:, :, 3);
gray = zeros(1080,1920);
for i = 1:1080
    for j = 1:1920
        gray(i,j) = max(max(r(i,j), g(i,j)), b(i,j));
    end
end    % Extract the maximum sub-pixel luminance matrix

V = mat2cell(gray, [repelem(40, 27)], [repmat(40, 1, 48)]);
X = cellfun(@max, cellfun(@max, V, 'UniformOutput', false));  % Calculate the initial backlight for each block
X = double(X);

% Get the size of the original image
[height, width, ~] = size(img);

% Define the size of each partition
partitionWidth = 40;
partitionHeight = 40;

% Calculate the number of rows and columns for partitions
numRows = floor(height / partitionHeight);
numCols = floor(width / partitionWidth);

% Initialize a cell array to store the partitions
imagePartitions = cell(numRows, numCols);

% Split the image and store each partition
for i = 1:numRows
    for j = 1:numCols
        % Calculate the position of each partition
        startX = (j - 1) * partitionWidth + 1;
        startY = (i - 1) * partitionHeight + 1;

        % Extract the partition
        partition = img(startY:startY + partitionHeight - 1, startX:startX + partitionWidth - 1, :);

        % Store the partition
        imagePartitions{i, j} = partition;
    end
end

% Weighted weight determination
S = rgb2gray(img_sal);
S_max = double(max(S(:)));
S = double(S);
S_normal = S ./ S_max;
% Determine the number and proportion of background pixels

S_T = 40 .* S_normal;
num_back = sum(S(:) == 0);
n_back = num_back / (1920 * 1080);
S_T(S_T == 0) = n_back;

% Initialize a cell array to store the partitions
imagePartitions1 = cell(numRows, numCols);

% Split the image and store each partition
for i = 1:numRows
    for j = 1:numCols
        % Calculate the position of each partition
        startX1 = (j - 1) * partitionWidth + 1;
        startY1 = (i - 1) * partitionHeight + 1;

        % Extract the partition
        partition1 = S_T(startY1:startY1 + partitionHeight - 1, startX1:startX1 + partitionWidth - 1, :);

        % Store the partition
        imagePartitions1{i, j} = partition1;
    end
end

target_PSNR = 35; % Target PSNR
target_MSE = (255^2) / (10^(target_PSNR / 10));
lambda_color = 3;
% Calculate the target Total Squared Error (TSE)
lambda = 1920 * 1080 * lambda_color;
target_TSE = target_MSE * lambda;

cumulative_TSE_T = 0; % Initialize cumulative TSE_T
Y = X ; % Initialize Y

while cumulative_TSE_T < target_TSE
    cumulative_TSE_T = 0; % Reset cumulative TSE_T for current Y value
    
    for row = 1:numRows
        for col = 1:numCols
            % Get the current partition and its corresponding S_T partition

            partition = imagePartitions{row, col};
            partition1 = imagePartitions1{row, col};

            [row_partition, col_partition, ~] = size(partition);
            z = zeros(row_partition, col_partition);
            partition = double(partition);
            r = partition(:, :, 1);
            g = partition(:, :, 2);
            b = partition(:, :, 3);

            TSE_r = sum(sum((r > Y(row, col)) .* (r - Y(row, col)).^2 .* partition1));
            TSE_g = sum(sum((g > Y(row, col)) .* (g - Y(row, col)).^2 .* partition1));
            TSE_b = sum(sum((b > Y(row, col)) .* (b - Y(row, col)).^2 .* partition1));
            TSE_T = TSE_r + TSE_g + TSE_b;

             % Accumulate TSE_T value
            cumulative_TSE_T = cumulative_TSE_T + TSE_T;
        end
    end
    
    % Reduce Y by 1
    Y = Y - 1;
    Y = max(Y, 0);
end

Idown = X - Y;
D = X - Idown;    % Final partition backlight value determination
Z = zeros(1080,1920);
row1 = 0:40:1080;   
row2 = row1;
row1 = row1 + 1;
col1 = 0:40:1920;   
col2 = col1;
col1 = col1 + 1;
for m = 1:27
    for n = 1:48
        Z(row1(m):row2(m + 1), col1(n):col2(n + 1)) = D(m, n);
    end
end           % Apply the 27×48 backlight matrix to the entire backlight part

%BMA smoothing
w = [0.06 0.11 0.06;0.08 0.38 0.08;0.06 0.11 0.06];
E = imfilter(D, w, 'conv', 'replicate', 'same');
L = imresize(E, 2, 'bilinear');
F = imfilter(L, w, 'conv', 'replicate', 'same');
G = imresize(F, 2, 'bilinear');
H = imfilter(G, w, 'conv', 'replicate', 'same');
I = imresize(H, [1080, 1920], 'bilinear');
%BMA backlight smoothing result

% Linear pixel compensation
a = 255 * ones(1080, 1920);
b = a ./ I;
R = double(img(:,:,1));
G = double(img(:,:,2));
B = double(img(:,:,3));
II(:,:,1) = R .* b;
II(:,:,2) = G .* b;
II(:,:,3) = B .* b;
II(II>255) = 255;  
% Overflow rate calculation
RR = R;
GG = G;
BB = B;
RR(RR<=255) = 0;
GG(GG<=255) = 0;
BB(BB<=255) = 0;
RGB = RR + GG + BB;
OF = sum(RGB(:) > 0);
OFR = (OF / (1920 * 1080)) * 100;
    
% Final dimming image
IMG(:,:,1) = II(:,:,1) .* I / 255;
IMG(:,:,2) = II(:,:,2) .* I / 255;
IMG(:,:,3) = II(:,:,3) .* I / 255;

img = double(img);

% Calculate PSNR
mse = immse(img, IMG);
psnr = 10 * log10((255^2) / mse);

% Calculate SSIM
ssim_value = ssim(img, IMG);

% Calculate PPR
p0 = sum(sum(a));
p1 = sum(sum(Z));
PPR = (p0 - p1) / p0 * 100;