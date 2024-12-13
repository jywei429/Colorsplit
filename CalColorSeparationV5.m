% Color Separation : Find the Averaged Hues-Perform Color Separation
%
%Version 1, 2009/5/26 
%Version 2,2010/04/08, funtional form for GUI
%Version 3,2010/4/15, add decorrelation stretch to improve the separation
%Version 3, 2010/4/19, exclude background pixels for better separation
%Version 3a,2010/5/13, test for using unified transform matrix and offset
%Version 3b,2010/5/14, add decorr function and use multiple images to
%                      calculate the color transform matrix( and offset),
%                      and save with settings file for future use
%Version 3c,2010/11/26,solve memory issue so that T of multiple images can be
%                      calculated (Note the CalColorSepT is combined to
%                      this function)
%Version 4c,2011/3/1, adjust max value setting
%Version 5, 2012/3/23, fix T calculation issue
function [T,offset]=CalColorSeparationV5(handles)

% read in info from handles
pathname=handles.IOdata.pathname;
filename=handles.IOdata.selected_file;

show=handles.settings.sep_show;
range=handles.settings.sep_BKGD_range; 
ThresholdI=handles.settings.sep_I_th;

n=size(filename,2);

% Image depth is assumed to be in 16 bit mode 
disp('image depth is assumed to be in 16 bit mode for each color channel!');
I_depth='uint16';
Imax_depth=double(intmax(I_depth));

targetMean=[];
targetSigma=[];
rowsubs=[];
colsubs=[];
useCorr=1;
All_sample_sum=zeros(1,3);
All_sample_npixels=zeros(1);
All_BB=zeros(3,3);
for index=1:n

    %
    % Read in the image file
    %

    if strfind(filename{index},'.mat')
        IData=whos('-file', [pathname filename{index}]);
        imagedata=load([pathname filename{index}]);
        I1=imagedata.(IData.name);
        imagedata=I1;
        clear I1;
    else
        imagedata=double(imread([pathname filename{index}]));
    end
    

        %Transform the color space from RGB to HSI
        cmap = rgb2hsv(imagedata/Imax_depth);
        H=cmap(:,:,1);
        S=cmap(:,:,2);
        I=cmap(:,:,3);
        H=H*2*pi; %change the values from [0,1] to [0,2*pi]
        
        %
        %Idnetify the pixels belongs to the particle
        %
        [counts,x]=imhist(S,20); %Calculate image histogram of saturation
        Sstd=std2(S);            %Calculate the variation of the saturation
        
        %Find the maximum counts in the saturation histogram and assign as the
        %averaged background saturation value Stri
        for i=1:length(counts)
            if counts(i)==max(counts)
                idx=i;
                Stri=x(idx);
            else
            end
        end
        %Find the pixels that has a saturation value close to the Stri and
        %defined as background pixels
        S_threshold=Stri+range*Sstd;
        
        %Find those three color overlap pixels by checking the intensity
        %values; those pixels should have very low intensity values
        [countI,bin_I]=imhist(nonzeros(I));
        cum_countI=cumsum(countI);
        I_count=(size(nonzeros(I),1)-cum_countI)/size(nonzeros(I),1);
        I_idx=size(find(I_count>ThresholdI),1);
        I_threshold=bin_I(I_idx);

        [rowsub,colsub]=find(~(S<S_threshold)|(I>I_threshold));
        [r c nbands] = size(imagedata);
        ind = sub2ind([r c], rowsub, colsub);
        
        [r c nbands] = size(imagedata);        % Save the shape
        npixels = r * c;                % Number of pixels
        A = reshape(imagedata,[npixels nbands]);     % Reshape to numPixels-by-numBands
        samples = A(ind,:);
        
        
        %%
        if show==1
            % gca;cla;
            % axes(handles.axes_ColorSepMain);title('Selected Samples');
            % hold on;
            % imagesc(imagedata.*repmat(double(~(S<S_threshold)|(I>I_threshold)/Imax_depth),[1,1,3])/255);     
        end




        % Calculate the transformation matrix for color separation based on calibration images
        % modified from MATLAB's decorr function
        % Decorrelation stretch for a multiband image of class double.
        
        [r nbands] = size(samples);        % Save the shape
        npixels = r ;                % Number of pixels
        samples = reshape(samples,[npixels nbands]);     % Reshape to numPixels-by-numBands
        
        if isempty(rowsubs)
            B = samples;
        else
            ind = sub2ind([r c], rowsubs, colsubs);
            B = samples(ind,:);
        end
        
        meanB = mean(B,1);        % Mean pixel value in each spectral band
        BB=B'*B;
        All_BB=BB+All_BB;
        All_sample_sum=npixels*meanB+All_sample_sum;
        All_sample_npixels=npixels+All_sample_npixels;
        
end
All_meanB=All_sample_sum/All_sample_npixels;
n = All_sample_npixels;%size(B,1);            % Equals npixels if rowsubs is empty
if n == 1
    cov = zeros(nbands);
else
    cov = (All_BB - (n * All_meanB') * All_meanB)/(n - 1);  % Sample covariance matrix
end
[T, offset]  = fitdecorrtrans(meanB, cov, useCorr, targetMean, targetSigma);


