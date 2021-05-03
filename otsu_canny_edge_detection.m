%My Implementation of the Canny Edge Detection w/ Adaptive
%Thresholding Technique via Otsu's Thresholding Method.
clear;
clc;
%% read in imgae and apply guassian blur
img = im2double(imread('cameraman.tif'));

% kernel_1 = fspecial('gaussian',[5,5],2);
% kernel_2 = fspecial('gaussian',[5,5],8);
% kernel_3 = fspecial('gaussian',[5,5],16);
% 
% gauss1 = imfilter(img, kernel_1);
% gauss2 = imfilter(img, kernel_2);
% gauss3 = imfilter(img, kernel_3);

gauss1 = imgaussfilt(img, 2);
gauss2 = imgaussfilt(img, 8);
gauss3 = imgaussfilt(img, 16);

%Display  the blurred images
figure(1)
sgtitle('The Blurred Images')
subplot(1,3,1)
imshow(gauss1)
title('sigma = 2')

subplot(1,3,2)
imshow(gauss2)
title('sigma = 8')

subplot(1,3,3)
imshow(gauss3)
title('sigma = 16')

%Get the Canny edge implementation results for each blurred image
r1 = otsu_canny_edge(gauss1,0.01, 0.05, 1);

r2 = otsu_canny_edge(gauss2,0.01, 0.05, 2);

r3 = otsu_canny_edge(gauss3,0.01, 0.05, 3);

%Display the implementation's final results
figure(6)
sgtitle('Thresholded Results')
subplot(1,3,1)
imshow(r1)
title('sigma = 2')

subplot(1,3,2)
imshow(r2)
title('sigma = 8')

subplot(1,3,3)
imshow(r3)
title('sigma = 16')

% %Now compare to Matlab's canny edge detection
canny1 = edge(gauss1, 'canny');
canny2 = edge(gauss2, 'canny');
canny3 = edge(gauss3, 'canny');

figure(7)
sgtitle("Matlab's Canny Edge Detection")

subplot(1,3,1)
imshow(canny1)
title('sigma = 2')

subplot(1,3,2)
imshow(canny2)
title('sigma = 8')

subplot(1,3,3)
imshow(canny3)
title('sigma = 16')
% 
% 
% %Creating the combined multiscaled image
% r_combined = r1 + r2 + r3;
% 
% figure(8)
% imshow(r_combined)
% title('Combining the edge maps')

%% canny edge detection function

function [g] = otsu_canny_edge(f, thrL, thrH, pos)
    %canny edge detection with adaptive thresholding function takes as 
        %input the grayscale image. The pos variale is used for
        %subplotting. thrL and thrH are used if using the double
        %thresholding piece of code instead of the otsu method section of
        %code.
 
    %get the horizontal and vertical gradients of the image
    %Sobel kernels have better noise-suppression (smoothing) characteristics makes them preferable
    %because noise suppression is an important issue when dealing with
    %derivatives according to the textbook
    kx=[-1 0 1; -2 0 2; -1 0 1];
    ky=[-1 -2 -1; 0 0 0; 1 2 1];
    gx=imfilter(f, kx);
    gy=imfilter(f, ky);
    
    %get the gradient magnitude
    g_mag = sqrt(gx.^2 + gy.^2);
    
    %get the edge direction of image
    g_direction = atan2(gy, gx) * (180.0/pi);
    
    %Display the gradients
    figure(pos + 1)
    sgtitle('Gradients')
    
    subplot(2,2,1)
    imshow(gx)
    title('Horizontal')
    
    subplot(2,2,2)
    imshow(gy)
    title('Vertical')
    
    subplot(2,2,3)
    imshow(g_mag)
    title("Magnitude")
    
    subplot(2,2,4)
    imshow(g_direction)
    title("Angle")
    
    % First, we approximate each angle in the matrix g_direction. 
    % The angles of the g_direction are rounded down or up to each of the
    % following angles: 0, 45, 90, or 135. The angles are rounded to within
    % 22.5 degrees, in the forward and reverse directions. 
    [rows, cols] = size(f);
    local_max = zeros(rows,cols);
    for row = 2:rows-1
        for col = 2:cols-1
            if((g_direction(row,col) > -22.5 && g_direction(row,col) < 22.5) || (g_direction(row,col) > 157.5 && g_direction(row,col) < -157.5))
                if(g_mag(row,col) > g_mag(row, col+1) && g_mag(row,col) > g_mag(row, col-1))
                    local_max(row,col)=g_mag(row,col);
                else
                    local_max(row,col)=0;
                end
            end
            if((g_direction(row,col) > 22.5 && g_direction(row,col) < 67.5) || (g_direction(row,col) > -157.5 && g_direction(row,col) < -112.5))
                if(g_mag(row,col) > g_mag(row+1, col+1) && g_mag(row,col) > g_mag(row-1, col-1))
                    local_max(row,col)=g_mag(row,col);
                else
                    local_max(row,col)=0;
                end
            end
            if((g_direction(row,col) > 67.5 && g_direction(row,col) < 112.5) || (g_direction(row,col) > -112.5 && g_direction(row,col) < -67.5))
                if(g_mag(row,col) > g_mag(row+1, col) && g_mag(row,col) > g_mag(row-1, col))
                    local_max(row,col)=g_mag(row,col);
                else
                    local_max(row,col)=0;
                end
            end
            if((g_direction(row,col) > 112.5 && g_direction(row,col) < 157.5) || (g_direction(row,col) > -67.5 && g_direction(row,col) < -22.5))
                if(g_mag(row,col) > g_mag(row-1, col+1) && g_mag(row,col) > g_mag(row+1, col-1))
                    local_max(row,col)=g_mag(row,col);
                else
                    local_max(row,col)=0;
                end
            end
        end
    end
    
%     figure(3)
%     imshow(approx_g_direction)
%     title('Approximated Gradient Direction')

    % Knowing the gradient direction of a pixel, the gradient magintude pixel is compared
    % to the neighbors along a normal to the gradient direction. If the
    % gradient magnitude is greater than both the neighbors, it is retained.
    % Otherwise, if the gradient magnitude of the pixel is set to zero. 

   
    figure(5)
    subplot(1,3,pos)
    imshow(local_max)
    sgtitle('nonmaximum suppression')
    
    %The below comment was for normal double thresholding. Uncomment this
    %section if you want to see those results
%     %Preforming double thresholding. If below lowest threshold, become 0.
%     %If above threshold, set to 1.
%     threshold_res = local_max;
%  
%      for i = 1  : row
%          for j = 1 : col
%              if (local_max(i, j) < thrL)
%                  threshold_res(i, j) = 0;
%              elseif (local_max(i, j) > thrH)
%                  threshold_res(i, j) = 1;
%              end
%          end
%      end
%      
%      g = threshold_res;
    
    %This section uses the Otsu Thresholding Method for Addaptive
    %Thresholding. Make sure the section above stays commented out if you
    %want to see the results of addpative thresholding
    local_max = im2uint8(local_max);
    g = otsu_thresholding(local_max);
    


end

%% The otsu method for addaptive thresholding
function [g] = otsu_thresholding(f)
    %takes in the input image and develops a histogram of the input image
        %It then uses the histogram to perform the otsu method to find the
        %thresholds based on the calculations obtatined using the histogram
        %of the image.
    [rows, cols] = size(f);
    n=imhist(f); % Compute the histogram
    N=sum(n); % sum up all the histogram values
    max=0; %initialize maximum 
    threshold = 0;
    P = zeros(256, 0);
    for i=1:256
        P(i)=n(i)/N; %Computing the probability of each intensity level
    end

    for T=1:255      % step through all thresholds from 1 to 255
        w0=sum(P(1:T)); %calculating the probability of class 1 (separated by threshold)
        w1=sum(P(T+1:256)); %calculating the probability of class 2 (separated by threshold)
        u0=dot((0:T-1),P(1:T))/w0; % calculating the class mean u0
        u1=dot((T:255),P(T+1:256))/w1; % calculating the class mean u1
        sigma=w0*w1*((u1-u0)^2); % compute sigma i.e  the variance between the class
        if sigma>max % compare sigma with maximum 
            max=sigma; % update the value of max i.e max=sigma
            threshold=T-1; % desired threshold corresponds to maximum variance of between class
        end
    end
    
    %Set thresholds
    th2 = threshold; 
    th1 = 0.5*th2;
    for i=1:rows
        for j=1:cols
            if (f(i,j) > th2)
                g(i,j)=255;
            elseif (f(i,j) < th1)
                g(i,j)=0;
            end    
        end
    end    
    

end


%Jamahl Savage 2021