old = imread('F:\Code\2019 Summer\Week2\test.jpg');
new = rgb2gray(old);
[rows, cols, colors] = size(old);
gray_pic = uint8(zeros(rows, cols));

for i = 1:rows
    for j = 1:cols
        sum = 0
        for k = 1:colors
            sum = sum + old(i, j, k) / 3
        end
        gray_pic(i, j) = sum;
    end
end

imwrite(gray_pic, 'F:\Code\2019 Summer\Week2\test2.jpg', 'png')

figure(1);
imshow(old)

figure(2)
imshow(new)

figure(3)
imshow(gray_pic)
         
