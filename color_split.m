clc
clear
close all

C4Image = imread('C4.tif');
PtImage = imread('PtTFPP2.tif');

% C4Image = im2double(C4Image);
% PtImage = im2double(PtImage);

% Step 2: Determine the size of the images
[sizeY_C4, sizeX_C4, ~] = size(C4Image);
[sizeY_Pt, sizeX_Pt, ~] = size(PtImage);

% Step 3: Define the cropping size
cropSize = 100;

% Step 4: Crop the middle portion of each image 
% Calculate the starting index for cropping for both images
startY_C4 = floor((sizeY_C4 - cropSize) / 2) + 1;
startX_C4 = floor((sizeX_C4 - cropSize) / 2) + 1;

startY_Pt = floor((sizeY_Pt - cropSize) / 2) + 1;
startX_Pt = floor((sizeX_Pt - cropSize) / 2) + 1;

% Crop the middle 100x100 portion from each image
C4Crop = C4Image(startY_C4:startY_C4 + cropSize - 1, startX_C4:startX_C4 + cropSize - 1, :);
PtCrop = PtImage(startY_Pt:startY_Pt + cropSize - 1, startX_Pt:startX_Pt + cropSize - 1, :);

% Step 5: Display the cropped images
% figure;
% subplot(1, 2, 1);
% imshow(C4Crop);
% title('Cropped Portion of C4 Image');
% 
% subplot(1, 2, 2);
% imshow(PtCrop);
% title('Cropped Portion of Pt Image');

C4Crop = C4Crop/2;
PtCrop = PtCrop/2;

% Step 6: Add the two cropped images together
sumImage = imadd(C4Crop, PtCrop);

C4Crop = C4Crop*2;
PtCrop = PtCrop*2;
% Step 7: Display the resulting image
figure;
imshow(sumImage);
title('Sum of Cropped Portions');

%% 
size = 100;
sigma = 50;
X = 1:size;
Y = X';
% G = 1 * exp(-1 / (sigma^2) * ((Y - size / 2).^2 + (X - size / 2).^2));
% GH = exp(-1 / (150^2) * ((Y - size / 2).^2 + (X - size / 2).^2));
G = ones(size, size);     
GH = ones(size, size);

% 1 green
H1 = .33*GH;
S1 = G;
V1 = G - .2;
hsvim1 = [];
hsvim1(:,:,1) = H1;
hsvim1(:,:,2) = S1;
hsvim1(:,:,3) = V1;
ref1 = hsv2rgb(hsvim1);

% 2 blue
ref2 = C4Crop;
ref2 = double(ref2);

ref2 = ref2 - min(ref2(:)); 
ref2 = ref2 / max(ref2(:)); 

% 3 red
ref3 = PtCrop;
ref3 = double(ref3);

ref3 = ref3 - min(ref3(:)); 
ref3 = ref3 / max(ref3(:)); 

% REFW
sum = ref1 + ref2 + ref3;
sumvector = reshape(sum, [], 3);
sumvector = min(max(sumvector, 0), 1); % [0,1]
refw = reshape(sumvector, size, size, 3);

% 
figure(1), imshow(ref1);
title('this is ref1');
figure(2), imshow(ref2);
title('this is ref2');
figure(3), imshow(ref3);
title('this is ref3');
figure(4), imshow(refw);
title('this is refw');

%%
% 
testim = (ref2 + ref3);
testim = testim - min(testim(:)); 
testim = testim / max(testim(:)); 

figure(), imshow(testim);
title('this is testim');

%% save
% currentFolder = pwd;
% fileName = 'test_img.tif';
% savePath = fullfile(currentFolder, fileName);
% imwrite(testim, savePath);
% fprintf('save path：%s\n', savePath);

%%
mw = refwpsp(refw);
mc1 = refcolorpsp(ref1);
mc2 = refcolorpsp(ref2);
mc3 = refcolorpsp(ref3);
data = psphsvtransform(testim, mc1, mc2, mc3, mw);

figure(), imshow(data);
title('this is data');
figure(), imshow(data(:,:,1));
figure(), imshow(data(:,:,2));
figure(), imshow(data(:,:,3));

function [mw] = refwpsp(image)
    Xrgb = reshape(image, [], 3);
    Xhsv = rgb2hsv(Xrgb);
    wvt = 0.4;
    wst = 0.1;
    Xwhite = [];
    for jj = 1:length(Xhsv)
        if Xhsv(jj,2) < wst && Xhsv(jj,3) > wvt
            Xwhite = [Xwhite; Xhsv(jj,:)];
        end
    end
    wpxs = hsv2rgb(Xwhite);
    mw = mean(wpxs);
end

function [meancolor] = refcolorpsp(image)
    Xrgb = reshape(image, [], 3);
    Xrgb = min(max(Xrgb, 0), 1); % [0,1]
    Xhsv = rgb2hsv(Xrgb);
    cvt = 0.05;
    cst = 0.4;
    Xcolor = [];
    for j = 1:length(Xhsv)
        if Xhsv(j,2) > cst && Xhsv(j,3) > cvt
            Xcolor = [Xcolor; Xhsv(j,:)];
        end
    end
    bincount = 4;
    [counts, edges] = histcounts(Xcolor(:,1), bincount);
    bincent = (edges(1:end-1) + edges(2:end)) / 2;
    [~, pkin] = max(counts);
    pkloc = bincent(pkin);
    margin = 0.05;
    colorpxs = Xcolor(abs(Xcolor(:,1) - pkloc) < margin, :);
    rgbcolorpxs = hsv2rgb(colorpxs);
    meancolor = mean(rgbcolorpxs, 1);
end

function [data] = psphsvtransform(data, rc1, rc2, rc3, rw)
    X = reshape(data, [], 3);
    T1 = arrayfun(@(i) det([X(i,:)', rc2', rc3']) / det([rw', rc2', rc3']), 1:size(X,1));
    T2 = arrayfun(@(i) det([rc1', X(i,:)', rc3']) / det([rc1', rw', rc3']), 1:size(X,1));
    T3 = arrayfun(@(i) det([rc1', rc2', X(i,:)']) / det([rc1', rc2', rw']), 1:size(X,1));
    T = [T1; T2; T3]';
    T = min(max(T, 0), 1); % [0,1]
    data = reshape(T, size(data,1), size(data,2), 3);
end

%%
f10 = figure();
f10.Position = [50 50 700 500];
subplot(2,3,1)
imagesc(sumImage(:,:,1))
title('R')
colorbar
colormap jet
pbaspect([1 1 1]) 
subplot(2,3,2)
imagesc(sumImage(:,:,2))
title('G')
colorbar
colormap jet
pbaspect([1 1 1])
subplot(2,3,3)
imagesc(sumImage(:,:,3))
title('B')
colorbar
colormap jet
pbaspect([1 1 1])

subplot(2,3,4)
imagesc(data(:,:,1))
title('G, Processed')
colorbar
colormap jet
pbaspect([1 1 1]) 
subplot(2,3,5)
imagesc(data(:,:,2))
title('B, Processed')
colorbar
colormap jet
pbaspect([1 1 1]) 
subplot(2,3,6)
imagesc(data(:,:,3))
title('R, Processed')
colorbar
colormap jet
pbaspect([1 1 1]) 

figure()
imagesc(data(:,:,2))
title('data(:,:,2)')

% %% Analyze data(:,:,2)
% % 
% ax1 = nexttile;
% hold on
% data_slice = data(:,:,2);
% 
% % 
% num_bins = 256;
% [counts, bin_edges] = histcounts(data_slice(:), num_bins);
% bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
% 
% plot(bin_centers, counts, 'DisplayName', 'data(:,:,2)')
% hold off
% grid on
% xlabel('Pixel Intensity')
% ylabel('Count')
% xlim([min(data_slice(:)), max(data_slice(:))])
% legend('Location', 'southoutside')
% 
% % data(:,:,2)
% ax2 = nexttile;
% imagesc(data_slice)
% colormap(gray)
% colorbar
% title('data(:,:,2)')
% xlabel('X-axis')
% ylabel('Y-axis')
% 
% linkaxes([ax1, ax2], "x")

% %% Analyze data(:,:,3)
% ax1 = nexttile;
% hold on
% data_slice = data(:,:,3);
% 
% %
% num_bins = 256; 
% [counts, bin_edges] = histcounts(data_slice(:), num_bins);
% 
% bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
% 
% plot(bin_centers, counts, 'DisplayName', 'data(:,:,3)')
% hold off
% grid on
% xlabel('Pixel Intensity')
% ylabel('Count')
% xlim([min(data_slice(:)), max(data_slice(:))])
% legend('Location', 'southoutside')
% 
% ax2 = nexttile;
% imagesc(data_slice)
% colormap(gray)
% colorbar
% title('data(:,:,3)')
% xlabel('X-axis')
% ylabel('Y-axis')
% 
% linkaxes([ax1, ax2], "x")


% Step 1: ref2 to HSV, ref2 is blue
ref2_hsv = rgb2hsv(ref2);
H = ref2_hsv(:,:,1); % Hue
S = ref2_hsv(:,:,2); % Saturation
V = ref2_hsv(:,:,3); % Value (Brightness)

% Step 2: adjust data(:,:,2) to match V
data_slice = data(:,:,2);
% data_scaled = mat2gray(data_slice); 
% data_scaled = imresize(data_scaled, size(V)); 
data_scaled = data_slice;


% Step 3: New HSV
new_hsv = cat(3, H, S, data_scaled);

% Step 4: Back to RGB
new_rgb = hsv2rgb(new_hsv);

% Step 5
figure;
subplot(1,3,1);
imshow(ref2);
title('Reference Image (ref2)');

subplot(1,3,2);
imshow(data_scaled);
title('Data Slice (Scaled)');

subplot(1,3,3);
imshow(new_rgb);
title('New Image with ref2 HSV characteristics');

%% Then do the same thing to Red

% Step 1: ref3 to HSV, ref3 is red
ref3_hsv = rgb2hsv(ref3);
H = ref3_hsv(:,:,1); % Hue
S = ref3_hsv(:,:,2); % Saturation
V = ref3_hsv(:,:,3); % Value (Brightness)

% Step 2: adjust data(:,:,2) to match V
data_slice = data(:,:,3);
% data_scaled = mat2gray(data_slice); 
% data_scaled = imresize(data_scaled, size(V)); 
data_scaled = data_slice;

% Step 3: New HSV
new_hsv = cat(3, H, S, data_scaled);

% Step 4: Back to RGB
new_rgb = hsv2rgb(new_hsv);

% Step 5: Plot
figure;
subplot(1,3,1);
imshow(ref3);
title('Reference Image (ref3)');

subplot(1,3,2);
imshow(data_scaled);
title('Data Slice (Scaled)');

subplot(1,3,3);
imshow(new_rgb);
title('New Image with ref3 HSV characteristics');
