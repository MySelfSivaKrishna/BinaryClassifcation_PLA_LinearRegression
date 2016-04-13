# BinaryClassifcation_PLA_LinearRegression
Binary Classification using PLA and Linear Regression                                     
you will implement the linear classification (PLA) and regression (pseudoinverse) algorithms covered in class. Create the following functions:

[X, Y] = generateData(N) which generates an (N x 2) matrix of points uniformly sampled from [−1, 1] × [−1, 1] and an (N x 1) vector of corresponding labels. Choose a random line in the plane as your target function f. Do this by taking two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the line passing through them, where one side of the line maps to +1 and the other maps to −1. Evaluate the target function on each xn to get the corresponding output yn.

[w, iters] = pla(X, Y, w0) returns a learned weight vector and number of iterations for the perceptron learning algorithm. w0 is the (optional) initial set of weights.

[w] = pseudoinverse(X, Y) returns the learned weight vector for the pseudoinverse algorithm for linear regression.

Using your implementations of each algorithm, we will compare the number of iterations it takes for PLA to converge with and without using Linear Regression to provide initial weights. For N = {10, 50, 100, 200, 500, 1000} compute the number of iterations of PLA with and without weight initialization using linear regression. For the uninitialized case, start the algorithm with the weight vector w being all zeros. Run each trial 100 times and compute the average for different randomly generated data sets. You should have a self-contained script, hw2.py, which performs all of the experiments for this assignment.

By the due date, turn in a ZIP file which contains:

Your Python source file named hw2.py which contains the three specified functions, a main function, plus any helper code you need.

A project write-up, which contains:

An English description of your algorithms, including any assumptions or design decisions you made.

Plot of the results of the experiments. You will be graded on how these results are presented.

Discussion of the various experiments and what contribution the various factors had on the number of iterations.

If there were any problems with your implementation (e.g. clearly wrong output) then make sure to indicate that in your write-up and give as much information as you can as to what you think is causing the problem.
