%This file contains the otsu canny edge detector as well as the recursive process 
clear;
clc;
%% read in image and apply guassian blur
img = im2double(imread('cameraman.tif'));

%recursive results using MATLAB's edge function
output1 = recursiveCanny(img, 128, 2);
output2 = recursiveCanny(img, 128, 8);
output3 = recursiveCanny(img, 128, 16);

figure(1)
sgtitle('Recursive Results')
subplot(1,3,1)
imshow(output1)
title('sigma = 2')

subplot(1,3,2)
imshow(output2)
title('sigma = 8')

subplot(1,3,3)
imshow(output3)
title('sigma = 16')

%recursive results using our otsu canny function
output1_2 = recursiveCanny2(img, 128, 2);
output2_2 = recursiveCanny2(img, 128, 8);
output3_2 = recursiveCanny2(img, 128, 16);

figure(2)
sgtitle('Recursive Results using Otsu')
subplot(1,3,1)
imshow(output1_2)
title('sigma = 2')

subplot(1,3,2)
imshow(output2_2)
title('sigma = 8')

subplot(1,3,3)
imshow(output3_2)
title('sigma = 16')

%blur images
gauss1 = imgaussfilt(img, 2);
gauss2 = imgaussfilt(img, 8);
gauss3 = imgaussfilt(img, 16);

%get results from our otsu canny edge detector
r1 = otsu_canny_edge(gauss1);
r2 = otsu_canny_edge(gauss2);
r3 = otsu_canny_edge(gauss3);

figure(3)
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

%MATLAB's function results
canny1 = edge(gauss1, 'canny');
canny2 = edge(gauss2, 'canny');
canny3 = edge(gauss3, 'canny');

figure(4)
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


%function for recursive canny, uses the MATLAB edge function to do canny
function output = recursiveCanny(img, baseCase, blur)
    %divide the image into 4 regions
    [region1, region2, region3, region4] = divideImage(img);    
    imgSize = size(img);
    %create our new outputs
    r1 = zeros(size(region1, 1), size(region1, 2));
    r2 = zeros(size(region2, 1), size(region2, 2));
    r3 = zeros(size(region3, 1), size(region3, 2));
    r4 = zeros(size(region4, 1), size(region4, 2));
    %for every region we created we check the size and if it is less than
    %or equal to the base case we filter the image and take the canny, if
    %not we simply call the recursive function again on the region
    for i = 1:4
        if(i == 1)
            if(size(region1, 1) <= baseCase)
                gauss = imgaussfilt(region1, blur);
                r1 = edge(gauss, 'canny');
            else
                r1 = recursiveCanny(region1, baseCase, blur);
            end
        elseif (i == 2)
            if(size(region2, 1) <= baseCase)
                gauss = imgaussfilt(region2, blur);
                r2 = edge(gauss, 'canny');
            else
                r2 = recursiveCanny(region2, baseCase, blur);
            end
        elseif (i == 3)
            if(size(region3, 1) <= baseCase)
                gauss = imgaussfilt(region3, blur);
                r3 = edge(gauss, 'canny');
            else
                r3 = recursiveCanny(region3, baseCase, blur);
            end
        elseif (i == 4)
            if(size(region4, 1) <= baseCase)
                gauss = imgaussfilt(region4, blur);
                r4 = edge(gauss, 'canny');
            else
                r4 = recursiveCanny(region4, baseCase, blur);
            end
        end          
    end 
    %join together our new regions for our ourput
    newSize = size(img, 1)/2;
    output = zeros(size(img, 1), size(img, 2));
    output(1:newSize, 1:newSize) = r1;
    output(newSize+1:imgSize(1), 1:newSize) = r3;
    output(1:newSize, newSize+1:imgSize(2)) = r2;
    output(newSize+1:imgSize(1), newSize+1:imgSize(2)) = r4;
    
end

%second recursive function, is identical to the above function but instead
%uses our canny edge detector function.
function output = recursiveCanny2(img, baseCase, blur)
    [region1, region2, region3, region4] = divideImage(img);    
    imgSize = size(img);
    r1 = zeros(size(region1, 1), size(region1, 2));
    r2 = zeros(size(region2, 1), size(region2, 2));
    r3 = zeros(size(region3, 1), size(region3, 2));
    r4 = zeros(size(region4, 1), size(region4, 2));
    for i = 1:4
        if(i == 1)
            if(size(region1, 1) <= baseCase)
                gauss = imgaussfilt(region1, blur);
                r1 = otsu_canny_edge(gauss);
            else
                r1 = recursiveCanny2(region1, baseCase, blur);
            end
        elseif (i == 2)
            if(size(region2, 1) <= baseCase)
                gauss = imgaussfilt(region2, blur);
                r2 = otsu_canny_edge(gauss);
            else
                r2 = recursiveCanny2(region2, baseCase, blur);
            end
        elseif (i == 3)
            if(size(region3, 1) <= baseCase)
                gauss = imgaussfilt(region3, blur);
                r3 = otsu_canny_edge(gauss);
            else
                r3 = recursiveCanny2(region3, baseCase, blur);
            end
        elseif (i == 4)
            if(size(region4, 1) <= baseCase)
                gauss = imgaussfilt(region4, blur);
                r4 = otsu_canny_edge(gauss);
            else
                r4 = recursiveCanny2(region4, baseCase, blur);
            end
        end          
    end 
    newSize = size(img, 1)/2;
    output = zeros(size(img, 1), size(img, 2));
    output(1:newSize, 1:newSize) = r1;
    output(newSize+1:imgSize(1), 1:newSize) = r3;
    output(1:newSize, newSize+1:imgSize(2)) = r2;
    output(newSize+1:imgSize(1), newSize+1:imgSize(2)) = r4;
    
end

%separate function to divide an image into 4 quadrants, used in the
%recursive functions
function [r1, r2, r3, r4] = divideImage(img)
    imgSize = size(img);
    newSize = imgSize(1)/2;
    r1 = img(1:newSize, 1:newSize);
    r3 = img(newSize+1:imgSize(1), 1:newSize);
    r2 = img(1:newSize, newSize+1:imgSize(2));
    r4 = img(newSize+1:imgSize(1), newSize+1:imgSize(2));
end


%% canny edge detection function

function [g] = otsu_canny_edge(f)
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
        
    % First, we approximate each angle in the matrix g_direction. 
    % The angles of the g_direction are rounded down or up to each of the
    % following angles: 0, 45, 90, or 135. The angles are rounded to within
    % 22.5 degrees, in the forward and reverse directions. 
    
    % Knowing the gradient direction of a pixel, the gradient magintude pixel is compared
    % to the neighbors along a normal to the gradient direction. If the
    % gradient magnitude is greater than both the neighbors, it is retained.
    % Otherwise, if the gradient magnitude of the pixel is set to zero. 
    
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
    
    %This section uses the Otsu Thresholding Method for Addaptive
    %Thresholding. Make sure the section above stays commented out if you
    %want to see the results of addpative thresholding
    local_max = im2uint8(local_max);
    g = otsu_thresholding(local_max);
end

%% The otsu method for addaptive thresholding
function [g] = otsu_thresholding(f)
    [rows, cols] = size(f);
    g = f;
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


%Jamahl Savage and Kevin Kleinegger 2021