fprintf('Implementing kNN...');

% MNIST data loading
train_images_file = fopen('train-images.idx3-ubyte', 'r', 'b');
train_labels_file = fopen('train-labels.idx1-ubyte', 'r', 'b');
test_images_file = fopen('t10k-images.idx3-ubyte', 'r', 'b');
test_labels_file = fopen('t10k-labels.idx1-ubyte', 'r', 'b');

% Extact header information 
fread(train_images_file, 4);
fread(train_labels_file, 4);
fread(test_images_file, 4);
fread(test_labels_file, 4);

train_size = fread(train_images_file, 1, 'int32');
test_size = fread(test_images_file, 1, 'int32');
num_rows = fread(train_images_file, 1, 'int32');
num_cols = fread(train_images_file, 1, 'int32');

% Data reading
train_images = fread(train_images_file, [num_rows*num_cols, train_size], 'uint8=>double')';
train_labels = fread(train_labels_file, train_size, 'uint8=>double');
test_images = fread(test_images_file, [num_rows*num_cols, test_size], 'uint8=>double')';
test_labels = fread(test_labels_file, test_size, 'uint8=>double');

% Data Preprocessing
train_images = train_images / 255;
test_images = test_images / 255;
num_classes = numel(unique(train_labels));


% Define k-value
k_values = [2 5 7 9 10 15];

%accuracy_array = zeros(1,6);

%for i = 1:6 ---> accuracy array inddices
for k = k_values

% Distance metric
distance_metric = 'euclidean';

% Initialize the predictions array
num_test = size(test_images, 1);
predictions = zeros(num_test, 1);

%num_classes = numel(unique(train_labels));
%confusion_matrix = zeros(num_classes,num_classes);

% Looping over test images
for i = 1:num_test
% Calculate the distance between the test image and all training images
%distances = pdist2(test_images(i,:), train_images, distance_metric); --> Euclidean Distance  
 
%distances = sqrt(sum((train_images - test_images(i,:)).^2, 2)); --> Euclidean Distance

distances=sum(abs(train_images - test_images(i,:)), 2) %--> Manhattan Distance

% Extracting indices of the k nearest neighbors
[~, indices] = mink(distances, k);

% Fetch labels of the k nearest neighbors
neighbor_labels = train_labels(indices);

% Finding most common label among the neighbors
prediction = mode(neighbor_labels);

% Adding predictions value to predictions array
predictions(i) = prediction;   
end



 % Updating the confusion matrix
%confusion_matrix(test_labels(i)+1, prediction+1) = confusion_matrix(test_labels(i)+1, prediction+1) + 1;


% Calculating accuracy of the predictions
accuracy = sum(predictions == test_labels) / num_test;
%disp(accuracy);
fprintf('Accuracy: %.2f%%\n', 100 * accuracy);
%accuracy_array(i) = accuracy;


C = confusionmat(10, predictions);
disp('Confusion matrix:');
disp(C);


end
%end
%disp(accuracy_array);

