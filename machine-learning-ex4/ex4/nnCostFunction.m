function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
origX = X;
X = [ones(m, 1) X];
outputs = diag (ones(1,num_labels));
for i = 1:m,
  costLayer1 = sigmoid(X(i,:)*Theta1');
  costLayer1 = [ones(size(costLayer1), 1) costLayer1];
  for j = 1:num_labels,
    costLayer2 = sigmoid(costLayer1*Theta2(j,:)');
    J = J - outputs(y(i),j)*log(costLayer2) - (1-outputs(y(i),j))*log(1-costLayer2);
  endfor
endfor

J = J/m;

correctionTheta1 = 0;
correctionTheta2 = 0;

for i = 1:size(Theta1,1),
  for j = 2:size(Theta1,2)
    correctionTheta1 = correctionTheta1+Theta1(i,j)^2;
  endfor
endfor
for i = 1:size(Theta2,1),
  for j = 2:size(Theta2,2)
    correctionTheta2 = correctionTheta2+Theta2(i,j)^2;
  endfor
endfor


correction = lambda*(correctionTheta1+correctionTheta2)/(2*m);
J = J+correction;

delta3 = zeros(size(Theta2,1),1);
costLayer3Vals = zeros(size(Theta2,1),1);
delta2 = zeros(size(Theta1,1),1);
for i = 1:m,
  costLayer2Vals = X(i,:)*Theta1';
  costLayer2 = sigmoid(costLayer2Vals);
  costLayer2Vals = [ones(size(costLayer2Vals), 1) costLayer2Vals];
  costLayer2 = [ones(size(costLayer2), 1) costLayer2];
  for j = 1:num_labels,
    costLayer3Vals(j) = costLayer2*Theta2(j,:)';
    costLayer3 = sigmoid(costLayer3Vals(j));
    %J = J - outputs(y(i),j)*log(costLayer2) - (1-outputs(y(i),j))*log(1-costLayer2);
    delta3(j) = costLayer3 - outputs(y(i),j);
  endfor
delta2 = Theta2'*delta3;
delta2 = delta2.*sigmoidGradient(costLayer2Vals');
delta2 = delta2(2:end);
Theta1_grad = Theta1_grad + delta2*(X(i,:));
Theta2_grad = Theta2_grad + delta3*(costLayer2);
endfor
Theta1_grad = Theta1_grad./m;
Theta2_grad = Theta2_grad./m;

p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = Theta1_grad + p1;
Theta2_grad = Theta2_grad + p2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
