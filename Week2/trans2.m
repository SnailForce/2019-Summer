img = imread('./frog2.jpg');
% figure(1);
% imshow(img);
img = imresize(img, [32, 32]);
% figure(2);
% imshow(img);
x = img;
x1 = abs(x);
x2 = reshape(x1, [1024, 1]);
t1 = min(x2);  % 53
t2 = max(x2);  % 214
x2 = double(x2);
x3 = (x2 - double(t1)) / double((t2 - t1));
r = x3
x3 = diag(x3);
x4 = diag(1./r);
x4 = x4 * 0.1 / 256;
k = cell2mat(f_set(1));
y = conv2(img, k, 'valid');
y = reshape(y, [400, 1]);
t3 = min(y);
t4 = max(y);
y = (y - t3) / (t4 - t3);
K = double(zeros(400, 1024));
for i = 1 : 13
    if i == 1
        disp(size(k(i)));
    end
    K(1, (i-1)*32+1:(i-1)*32+13) = k((i-1)*13+1 : (i-1)*13+13);
end

