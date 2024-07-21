function [rmsvars, lowndx, rmstrain, rmstest] = a2_20292366
% [RMSVARS LOWNDX RMSTRAIN RMSTEST]=A3 finds the RMS errors of
% linear regression of the data in the file "GOODS.CSV" by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. For the variable that is
% best explained by the other variables, a 5-fold cross validation is
% computed. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         none
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

% Imports goods.csv as filename and calls functions a2q1 and a2q2
filename = 'goods.csv';
[rmsvars, lowndx] = a2q1(filename);
[rmstrain, rmstest] = a2q2(filename, lowndx)

end

function [rmsvars, lowndx] = a2q1(filename)
% [RMSVARS LOWNDX]=A2Q1(FILENAME) finds the RMS errors of
% linear regression of the data in the file FILENAME by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX.
%
% INPUTS:
%         FILENAME - character string, name of file to be processed;
%                    assume that the first row describes the data variables
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS

% Reads data from goods.csv, skipping first row and column
% Finds size of dataset

data = csvread(filename, 1, 1);
[m, n] = size(data);

% Creates 1*16 vector rmsvars that stores the rms error values for each 
% type of commodity in goods.csv, lowest of which is represents the best 
% fitting variable/feature to a linear regression model as the model's 
% predictions are very close to the actual values to a relatively high 
% accuracy.

rmsvars = zeros(1, n);

% We iterate through all the columns in the dataset, treating each one as a
% dependent variable and the rest as independent variables. The Root Mean
% Square Error for each instance is calculated and stored in rmsvars. 

for i = 1:n
    % Extracts data from each column to treat as a dependent variable in y_vec
    % Extracts data from the rest of the columns to produce a design matrix
    % in X_mat

    % Note that the first column in X_mat is a column of ones: This acts as
    % an intercept term that allows the line produced by the model to
    % intercept the axes at the origin, even if the actual model between
    % the independent and dependent variables doesn't organically have
    % that. The purpose of this is discussed in the discussions section in 
    % the report.

    y_vec = data(:, i);
    X_mat = [ones(m, 1), data(:, [1:i-1, i+1:n])];

    % The linear regression between the dependent variable vector y_vec and
    % the independent variable matrix X_mat is calculated below using the \
    % operator. The predicted values for each dependent variable is
    % calculated using the independent variable matrix and the weight vector
    % calculated by doing a regression.
    
    w_vec = (X_mat' * X_mat) \ (X_mat' * y_vec);
    y_pred = X_mat * w_vec;

    % The RMS Error values between the actual and predicted data points are
    % then concurrently calculated and stored in rmsvars.
    
    rmsvars(i) = rms(y_vec - y_pred);
end

% Use of the min function on the dependent variable vector is initiated to
% extract the index of the commodity that had the best fitting actual values
% to their predicted values in the regression model, ie. the commidity with
% the lowest RMS error between actual and predicted values.

[~, lowndx] = min(rmsvars);

% The commodity best modelled by the linear regression, Uranium in our case,
% is then initialized to be graphed. y_vec stores the actual values for the 
% good, X_mat the independent variable matrix, w_vec the weight vector from
% the linear regression, and y_pred the predicted values for our particular
% good.

y_vec = data(:, lowndx);
X_mat = [ones(m, 1), data(:, [1:lowndx-1, lowndx+1:n])];
w_vec = (X_mat' * X_mat) \ (X_mat' * y_vec);
y_pred = X_mat * w_vec;

% The plot function then plots the actual datapoints from the given dataset
% for Uranium using the argument 'o' for circular marks. The 'hold on'
% command allows the predicted values to be shown in the form of a
% regression on the sample plot using the argument 'r'. The rest of the
% code below handles the output which makes the graph readable.

plot(y_vec, 'o');
hold on;
plot(y_pred, 'r');
xlabel('Quantity in arbitrary units');
ylabel('Price in $/unit');
title(sprintf('Regression for Column %d with Lowest RMS Error for best modelled commodity - Uranium', lowndx));
legend({'Observed Values', 'Predicted Values'});

end

function [rmstrain, rmstest] = a2q2(filename,lowndx)

% Reads data from goods.csv, skipping first row and column
% Finds size of dataset
data = csvread(filename, 1, 1);
[m, n] = size(data);

% X_mat and y_vec store the independent variable matrix and the dependent
% variable vector respectively, where lowndx is the index of the commodity
% with the lowest RMS Error value.

X_mat = data(:, [1:lowndx-1, lowndx+1:n]);
y_vec = data(:, lowndx);

% This resets the random number generator to ensure that any future calls
% to random number generation functions, ie. randidx, will output the same
% sequence every time the function is called. This helps us moderate,
% compare and interpret results of changes in the overall code.
% The randidx variable stores random order of numbers generated by the 
% randperm function which is later used to randomly select a subset of
% indices from a vector.

rng('default');
randidx = randperm(m);

% rmsTrain and rmsTest vectors are initialized to store the output
% generated by running a RMS Error analysis on the linear regressions
% performed on the 5 different folds created from the data in Xmat above.
% foldSize calculates the size of each fold using the floor function
% rounding down to the closest integer which creates equal sized folds,
% with a minimal number of rows being extra in case of imperfect division.

rmsTrain = zeros(1,5);
rmsTest = zeros(1,5);
foldSize = floor(m / 5);

% We iterate through 5 folds to find RMS Error values for the linear regression 
% training and testing models

for i = 1:5

    % StartIdx and endIdx calculate the starting and ending indices for the
    % fold in iteration respectively. testIdx stores randomly selected
    % indices to form a testing set using randidx. trainIdx is the direct
    % complement of testIdx which takes in all the independent variables in
    % the X_mat
    
    startIdx = (i - 1) * foldSize + 1;
    endIdx = startIdx + foldSize - 1;
    testIdx = randidx(startIdx:endIdx);
    trainIdx = setdiff(1:m, testIdx);

    % These are training and test sets for X_mat and y_vec using the
    % trainIdx and testIdx statements described above.
    
    Xtrain = X_mat(trainIdx,:);
    ytrain = y_vec(trainIdx);
    Xtest = X_mat(testIdx,:);
    ytest = y_vec(testIdx);
    
    % Calculation of the w_vec weight vector as a result of the linear
    % regression. The predicted values for training and testing sets are
    % calculated using the regression coefficient and the X_mat adjusted
    % sets for training and testing.

    w_vec = Xtrain\ytrain;
    ypredtrain = Xtrain*w_vec;
    ypredtest = Xtest*w_vec;
    
    % rmsTrain and rmsTest are vectors that store the RMS Error values
    % between the actual and predicted values for both the training and
    % testing datasets.

    rmsTrain(i) = rms(ytrain - ypredtrain);
    rmsTest(i) = rms(ytest - ypredtest);


end

% rmstrain and rmstest are assigned rmsTrain and rmsTest to be output.

rmstrain = rmsTrain;
rmstest = rmsTest;

end

function [rmstrain,rmstest]=mykfold(Xmat, yvec, k_in)
% [RMSTRAIN,RMSTEST]=MYKFOLD(XMAT,yvec,K) performs a k-fold validation
% of the least-squares linear fit of yvec to XMAT. If K is omitted,
% the default is 5.
%
% INPUTS:
%         XMAT     - MxN data vector
%         yvec     - Mx1 data vector
%         K        - positive integer, number of folds to use
% OUTPUTS:
%         RMSTRAIN - 1xK vector of RMS error of the training fits
%         RMSTEST  - 1xK vector of RMS error of the testing fits

% Problem size
M = size(Xmat, 1);

% Set the number of folds; must be 1<k<M
if nargin >= 3 && ~isempty(k_in)
    k = max(min(round(k_in), M-1), 2);
else
    k = 5;
end

% Initialize the return variables
rmstrain = zeros(1, k);
rmstest  = zeros(1, k);

% Process each fold
for ix=1:k
    % %
    % % STUDENT CODE GOES HERE: replace the next 5 lines with code to
    % % (1) set up the "train" and "test" indexing for "xmat" and "yvec"
    % % (2) use the indexing to set up the "train" and "test" data
    % % (3) compute "wvec" for the training data
    % %
    xmat_train  = [0 1];
    yvec_train  = 0;
    wvec = [0 0];
    xmat_test = [0 1];
    yvec_test = 0;

    rmstrain(ix) = rms(xmat_train*wvec - yvec_train);
    rmstest(ix)  = rms(xmat_test*wvec  - yvec_test);

end

end