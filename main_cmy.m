clc;
clear;
close all;

% parameters
handles.IOdata.pathname = './';
handles.IOdata.selected_file = {'cmy_test.tif'}; % List of image files
handles.settings.sep_show = 1; % Display intermediate results
handles.settings.sep_BKGD_range = 0.2; % Range parameter for background pixels
handles.settings.sep_I_th = 0.1; % Intensity threshold

% color separation transformation matrix and offset
[T, offset] = CalColorSeparationV5(handles);

% Process each image and extract CMY channels
for idx = 1:length(handles.IOdata.selected_file)
    % current image
    img_path = fullfile(handles.IOdata.pathname, handles.IOdata.selected_file{idx});
    img = double(imread(img_path)); 

    % Check if the image is RGB
    if size(img, 3) ~= 3
        error('Input image must be an RGB image');
    end

    % Apply the color separation transformation
    transformed_image = ApplyColorSeparation(img, T, offset);

    figure();
    imshow(transformed_image / max(transformed_image(:)));
    title('transformed image');

    % CMY channels
    cyan_channel = 1 - transformed_image(:, :, 1);  % Cyan = 1 - Red
    magenta_channel = 1 - transformed_image(:, :, 2); % Magenta = 1- Green
    yellow_channel = 1 - transformed_image(:, :, 3);  % Yellow = 1-Blue

    % Normalize each channel
    cyan_channel = (cyan_channel - min(cyan_channel(:))) / (max(cyan_channel(:)) - min(cyan_channel(:)));
    magenta_channel = (magenta_channel - min(magenta_channel(:))) / (max(magenta_channel(:)) - min(magenta_channel(:)));
    yellow_channel = (yellow_channel - min(yellow_channel(:))) / (max(yellow_channel(:)) - min(yellow_channel(:)));

    % Display
    figure;
    subplot(3, 4, 1);
    imshow(img / max(img(:)), []); % Original image
    title(['Original Image: ', handles.IOdata.selected_file{idx}]);

    subplot(3, 4, 2);
    imshow(cyan_channel, []);
    title('Cyan Channel');

    subplot(3, 4, 3);
    imshow(magenta_channel, []);
    title('Magenta Channel');

    subplot(3, 4, 4);
    imshow(yellow_channel, []);
    title('Yellow Channel');

    % heatmaps for each channel
    subplot(3, 4, 6);
    imagesc(cyan_channel);
    colormap('jet');
    colorbar;
    title('Cyan Channel Heatmap');

    subplot(3, 4, 7);
    imagesc(magenta_channel);
    colormap('jet');
    colorbar;
    title('Magenta Channel Heatmap');

    subplot(3, 4, 8);
    imagesc(yellow_channel);
    colormap('jet');
    colorbar;
    title('Yellow Channel Heatmap');

    % intensity overlay heatmap
    intensity_image = cyan_channel + magenta_channel + yellow_channel;

    subplot(3, 4, 10);
    imagesc(intensity_image);
    colormap('jet');
    colorbar;
    title('Intensity Heatmap');
end

% ApplyColorSeparation
function transformed_image = ApplyColorSeparation(image, T, offset)

    % Reshape
    [rows, cols, channels] = size(image);
    if channels ~= 3
        error('Input image must be an RGB image');
    end
    image_reshaped = reshape(image, [], 3); % Reshape to an N×3 matrix

    % Apply the color separation transformation
    transformed = (image_reshaped - offset) * T'; % Linear transformation

    % Reshape the transformed data back to image dimensions
    transformed_image = reshape(transformed, rows, cols, 3);
end
