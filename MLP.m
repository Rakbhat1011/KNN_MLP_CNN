     trainSet =  loadMNISTImages('train-images.idx3-ubyte');
     tarinLabel =  loadMNISTLabels('train-labels.idx1-ubyte');
     testSet = loadMNISTImages('t10k-images.idx3-ubyte');  
     testLabel = loadMNISTLabels('t10k-labels.idx1-ubyte');

     training_Size = size(trainSet);  
     testing_Size = size(testSet);  

     sum = training_Size(2);  
     hidden_units = 250;  
     input_units = training_Size(1);  
     output_units = 10;  
     num_of_iterations = 5;  

     %Initialize the weights from Uniform(0, 1)  

     %Layer-1 weights
     weight_1 = rand(input_units + 1, hidden_units);
     weight_2 = rand(hidden_units + 1, output_units); 

      % Scaling factor in sigmoid function
     scale_factor = 0.01;  
     % Learning rate 
     learning_rate = 0.1; 


     %Matrix to store weight updates and activations  
     f1 = zeros(input_units + 1, 1);  
     f2 = zeros(hidden_units + 1, 1);  
    
     f22 = zeros(hidden_units, 1);  
     f3 = zeros(output_units, 1); 

     error_1 = zeros(input_units + 1, 1);  
     error_2 = zeros(hidden_units + 1, 1);  
     error_3 = zeros(output_units, 1);  

    temp = zeros(output_units);
    temp(1:(output_units+1):end) = 1;
 
     correct = zeros(num_of_iterations, 1);  

     fprintf('\n Implementing MultiLayer Perceptron..\n');

     %Training  
     for k = 1:num_of_iterations  
       fprintf('\n%d Loop: \n', k);
       fprintf('Data points covered: ');

       % Shuffle data points randomly
       perm = randperm(sum);  
       for p = 1:sum  
         % Print progress message
         if (mod(p, 1250) == 0)  
           fprintf('\t%d', p);  
         end 

         % Forward Propogation 
         index = perm(p);  
         f1 = [trainSet(:, index); 1];  
         f22 = sigmfn(weight_1' * f1, [scale_factor, 0]);  
         f2 = [f22; 1];  
         f3 = sigmfn(weight_2' * f2, [scale_factor, 0]); 

         % Backward propogation 
         error_3 = f3 - temp(:, tarinLabel(index) + 1);  
         error_2 = weight_2 * (error_3 .* f3 .* (1 - f3));  
         error_2 = error_2(1:hidden_units);  
         error_1 = weight_1 * error_2;  

         % Weight Updation  
         weight_2 = weight_2 - learning_rate * f2 * (error_3 .* f3 .* (1 - f3))';  
         weight_1 = weight_1 - learning_rate * f1 * (error_2 .* f22 .* (1 - f22))';  
       end  

       %Check for error in training  
       correct(k) = 0;  
       for i = 1:sum  
         f1 = [trainSet(:, i); 1];  
         f2 = [sigmfn(weight_1' * f1, [scale_factor, 0]); 1]; 
         f3 = sigmfn(weight_2' * f2, [scale_factor, 0]);  
         [~, m] = max(f3);  
         if (m == tarinLabel(i) + 1)  
           correct(k) = correct(k) + 1;  
         end  
        end  
     end  


    %Calculate Accuracy
    trainingAccuracy = (correct /  training_Size(2))*100;
     successTest = 0;  

     for i = 1:testing_Size(2)  
       f1 = [testSet(:, i); 1];  
       f2 = [sigmfn(weight_1' * f1, [scale_factor, 0]); 1];  
       f3 = sigmfn(weight_2' * f2, [scale_factor, 0]);  
       [~, m] = max(f3);  
       if (m == testLabel(i) + 1)  
         successTest = successTest + 1;  
       end 
     end 

     testingAccuarcy = (successTest /  testing_Size(2))*100;
    fprintf('\nAccuracy = %f\n', testingAccuarcy);


    %Calculate Confusion Matrix
    confusion_matrix = zeros(output_units);
     fprintf('\nConfusion Matrix:\n');

for i = 1:testing_Size(2)
    f1 = [testSet(:, i); 1];
    f2 = [sigmfn(weight_1' * f1, [scale_factor, 0]); 1];
    f3 = sigmfn(weight_2' * f2, [scale_factor, 0]);
    [~, predicted_label] = max(f3);
    true_label = testLabel(i) + 1;
    confusion_matrix(true_label, predicted_label) = confusion_matrix(true_label, predicted_label) + 1;
end

disp(confusion_matrix);



%--------------------------------------------------------------------------

function y = sigmfn(x, params)

    % Extract parameters
    a = params(1);
    c = params(2);

    % Compute sigmoid function
    y = 1 ./ (1 + exp(-a * (x - c)));
end


