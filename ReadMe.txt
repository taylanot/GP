(c) Ozgur Taylan TURAN 2019, Dec. 4th
- This scripts can be used to find a surrogate function for your data and could give you information where your next data should be to find the maximum of your function. 
------------------------------------------------------------------------------------------
* SOON TO BE DONE:
- To increase the performance try to reduce the number of restarts in acq.func. minimization, by using more equally spaced starting points instead of random starts.
- Plotting options should be polished it is just a rough visualization.
- The bounds creation for the minimization is a bit brute try a more elegent solution.
- You could add more acquisition functions to be utilized.
- Found a problem when the mazimum of the function is on the boundaries this code is having difficulties with proposing the new point.
