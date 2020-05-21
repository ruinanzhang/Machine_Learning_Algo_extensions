
# CS 6923 Machine Learning Final Project Report
***Group members:*** 

*Ruinan Zhang [rz1109@nyu.edu](mailto:rz1109@nyu.edu)*  

*Xianbo Gao  [xg656@nyu.edu](mailto:xg656@nyu.edu)*

## 1 Introduction and Report Structure:
### 1.1 Introduction
In this project, we explored three ML algorithms we learned during the pass of this course, for each basic algorithm, we came up with one extension that can improve the overall performance of that algorithm. And then, we first implemented the basic and extended algorithms using existing libraries (Scikit-learn, Tensorflow,Pytorch...). Second, we built our own extension only Numpy from scratch.

The three algorithms we explored and corresponding extensions are:
	1.  Support vector machine (SVM) with Soft Margin
	2.  Linear Regression vs. Lasso Regression with Coordinate descent 
	3.  Neural Network with Convolutional layers
## 2 Support vector machine (SVM) with Soft Margin

### 2.1 Explanation about SVM with soft margin algorithm

In homework9, we implemented hard margin SVM and solved both primal and dual problems. In this project, we will focus on solving soft margin dual problems as most datasets in real life are not linearly separable.

Remember in hard margin SVM case, constrained optimization problem (primal) is:

<img src="https://lh5.googleusercontent.com/rLNbU-mjSbvoBp_7eTEJDn0miqPHQDMtT26uDcdBejwEhBztTbdpP4Ux7ZBPWBvf6trqxPOXB-a0bG7jfsXVc2rX7GiMexiXqv4pjR81szQQPvDa1OzP2ZtdrvpntiXA2Xgt771z" width="20%" height="20%">

And after we transform our original problem into its duel formalization, adding Lagrangian multiplier, we have a dual optimization problem that is:

<img src="https://lh3.googleusercontent.com/FsZ47XAaZFORuUXAD-u75Fsv60AbNBqqb3dgxh9BP1FqVlM6FomSw_aqRgJtmo8_0cWF1IA68shzeu9S2CIog_xUpCiUYnyF7ohO8vAoxRMP4c1S1lfF-tV0mEKHhOyNSAGDfolg" width="20%" height="20%">

The hard margin SVM has its limit as it assumes that our data is linearly separable. But what if we have a few misclassified data or data points that within the margin?

Then we extend to the soft margin SVM case -- which introduces a new tolerance variable (or slack variable) 𝜁(𝑖) for each training example 𝑥(𝑖).

<img src="https://lh5.googleusercontent.com/ztRk4JuMh5u_RlnfhDVc5ix6WUPRNJwPaPPsXNB1W2u6slS9xsWVQILGByv2LMyRcWo7uVhqXgT0t8OXef9ZKMgNngGwHhatHybB6L1sXWrltucsV-38rKOZHyH4T_5L3VCPjDpd" width="60%" height="60%">

[[1] Pic Sourc](https://sandipanweb.wordpress.com/2018/04/23/implementing-a-soft-margin-kernelized-support-vector-machine-binary-classifier-with-quadratic-programming-in-r-and-python/)

Points that lie on the margin will have 𝜁(𝑖)=0. On the other hand, the farther we are from the correct boundary, the larger the value of 𝜁(𝑖) will be larger.

Thus, our optimization function would be:

<img src="https://lh5.googleusercontent.com/1JvvyhoVsqFos56BT0OLtPV1t1bdtBBIXcFDlIwA4ccCJy86_yV3ZuOudNDGJeA_jgr-FMlyk-sYnZfalR1AGGoW-jaTIKX26hHSWlGToO9R1zHtKohYZPCcddwjPu4cHSHirxEB" width="20%" height="20%">

Notice here C is a tunable parameter, gives relative importance of the error term. It controls how much you want to punish your model for each misclassified point for a given curve. The lower the C parameters, the softer the margin.

After adding the Lagrangian multiplier, to do the kernel trick, then we have the SVM soft margin dual quadratic programming problem:

<img src="https://lh4.googleusercontent.com/cFEw4m4ciJdCts1bBWJxeDkjJ-coxqrgXh0sgb8byymPz4b0kVAULQZzHF8KIFn-t3IJK9ZjLJl70MQt1d1IyD5yWpQcV8LZ4AmHmYke1HzGCOKDOfVxU2H8aTPWpaZcPZaDxdl9" width="20%" height="20%">

Then, we use CVXOPT, the quadratic programming problem solver, cvxopt.solvers.qp to solve this problem. We can see this is very similar to the homework's problem of dual for hard margin besides we have an additional constraint on 𝛼.

When using CVXOPT's quadratic solver, we implement this constraint by concatenating below matrix G a diagonal matrix of 1s of size m×m and adding m times value to C in vector h.

### 2.2 Summary SVM’s soft margin’s implementation using Sklearn and Numpy

-   Datasets: 

For SVM, we chose to run this algorithm on two datasets: (1) the Iris dataset  used in hw3 (2)Breast Cancer Wisconsin (Diagnostic) dataset

-   Hard Margin and Soft Margin SVMs using Sklearn
    

![](https://lh5.googleusercontent.com/krrHIY6GFCopS1HCTE27OFq4N121FACuWFF5kPbmXnetoZxxoArfd4D55MfTUfxkoE9lBTTaaV3VvYQ-KLzQE0Jjuy_4HLi3cBS_SZGME8YazpDS-NUP-gltprFuemU0NJmHCiIX)

In Sklearn’s SVM, we can see it's using C as regularization parameter that controls how much you want to punish your model for each misclassified point for a given curve. The lower the C parameters, the softer the margin.

Since we used linear kernel for SVM during the class, we also chose Linear Kernel for the Iris dataset. However, to explore soft margin’s effect, we also used RBF kernel on the other dataset (Breast Cancer db)

For hard margin SVM, using linear Kernel we set param C to 1000 and for soft margin SV, we set param C to 1.5 for both data sets mentioned above.

-   Soft Margin SVM built using Numpy

We also used C =1.5 and Linear Kernel when we implemented our own Soft margin extension on SVM as shown below:

![](https://lh3.googleusercontent.com/NrBZZRxco1PJh2LkJBbdMa_WHGMr4lmRbK7IhFuguQ0IJ8p25jwVz7C7VSvNXH10DdmRJt9eevRj-rxP5NSE9B56NoWj49HprBS2x_ACcQz2J7SR6qLOIBvgKLLml9HXe8qAQ7CX)

### 2.3 Accuracies Table for SVM

Note: since both Iris dataset and Breast cancer dataset are relatively small data set, we used cross-validation function from Sklearn for calculating the accuracies: 

<img src="https://lh6.googleusercontent.com/nVNAtzpFNY9gHgvF32H_Ac4TVtXlSfc61mqKZH5K7_udCtxZrMKPj2GzsEneB1jPzJzBLOue7UQagsxHc3_9d6rMwj1wRZayOmqSPvWP0wlwRTWAXilL3Xf1ggPe_POI8bqPcZ-O" width="60%" height="60%">

### 2.4 Explanations on specific datasets for SVM

-   Iris Data
Using Sklearn, we can see some improvement that for each cross validation set:

<img src="https://lh3.googleusercontent.com/N3fN2aD7nQNkGkISRvwX0FRjkba50JojqnPI0bjWtLuAwX2CxKv63u9kcXqrV5RM8NBy-yE3b89A4vjhhJvnLfHubWeqX5-sImx1BnlLbPI95DAASc7t4BKB6jxNKJAYST7UDuV4" width="60%" height="60%">

In our own Numpy implementation, we analyzed how accuracy might improve on Iris using different param C values:

<img src="https://lh4.googleusercontent.com/py8Lh5pYoLyqfkXnwE3qkCm6Tq4GZezQDn4Ym6PkXYYZgu4d5VlUlDFSaMd8f5xeEnze0YG-BOUr-rXOWzu8UCoHINxG8FXWxzc9MC-rySQ3uH6ndjB__5-F8E-udLPvFSQdSTAG" width="60%" height="60%">

-   Breast Cancer Dataset
    

Using Sklearn, we can see some improvement that for each cross validation set:

<img src="https://lh4.googleusercontent.com/TXUcn_Hez-AuYro8kq0zly27AjLXFUbprbtg4YRl2jHZQbbbkbAyvzaX9F5EbLHyNiUdJTcX6kWramwGODoNAbcD55AqGUx5Bc8kb1VjJHIofKONkdTXgrok1B5GIxshrDbk8di9" width="60%" height="60%">

In our own Numpy implementation, we analyzed how accuracy might improve on Brest Cancer Dataset using different param C values:

<img src="https://lh3.googleusercontent.com/FvjY_4EvrZMVIYX7A26eTM3B48g9Iru9_KaKhOlyA_66zvQIjDN51ZLGJYnmTndKmxWObpmrFyJ986ybSrb51Rr0RHT9RsoaNbc7U7V5RwWgg4QktjkCqNnoBDSBXrkdPMk_qqWz" width="60%" height="60%">

#### From the chart above, different from the Iris data set, we can see that the accuracy first goes up and then goes down as we increase the C value. One possible explanation is at the beginning we have very small penalty so that there's an under fitting problem

### 2.5 Numpy Code for soft margin SVM

![](https://lh5.googleusercontent.com/dHILuTc3T1DMagjZATwsbLgglHBpryhUlLvyLm82cmwwjXdZEEtpCAfmCqJgjDKPL5UKBO7DAYZUxiUq2gB2gICFcuhhrZA_x3DGhyXVYircwsuZvSlJniKkkIojo6rm96Atb-AU)

![](https://lh5.googleusercontent.com/A9gl6JBPGZO80kIuwCufjnSouXCOjBKt1p_vT_z3k9nOySIh6cdHiaghkUbF7J2N0HuDv292z9sS4rWOALMmtVDoQblD9h7qQdbvrE-B5XABbQ1CCIMoO2GX95nZxY4z1l2QV4bu)

![](https://lh6.googleusercontent.com/wsgY9ymS1jldb-6HxgFbph8T8pppFF9zJSRVGzRszkbdI8qZ4AVa3dyj3X6PAbxSNdvbWxEc9t18bGe6pmTLHpEuLbEpn3BT7phQOyswDUNVfoL9uuN4OQP5j1h7NsZPwec5xEvT)


## 3. Lasso Regression with Coordinate descent

### 3.1 Explanation about Lasso with coordinate descent

In class, we implemented the Ridge Regression with L2 regularization. In the lecture, the professor also covered Lasso regression with L1 regularization. Not like Ridge Regression that has a closed form solution, the Lasso Regression has no closed form solution to minimize :

<img src="https://lh6.googleusercontent.com/BZ43YGQ6CftlaA4AGGoEXW7JEmnUWLNEdEdEv2FOi7LOAmZjfplOZZI5wemc4j5SQr7ZBm9q412FcMUyiMKJiMO_fcZV5EDfybiu-5jO8MH87rtgOlk996oWGZjplxRFRDK0VAN-" width="20%" height="20%">

Although 𝐸𝑙𝑎𝑠𝑠𝑜 (𝑤) is convex, we can not take the derivative and set it to zero to find the optimal 𝑤. However, it's possible to optimize the 𝑗𝑡ℎ coefficient while the others remain fixed:

<img src="https://lh4.googleusercontent.com/qnDE8UnH8cI3fD9IazPkBV5-Ysh_XyCm7GXOTdSMLoeaQrjsf1gUHEA4--U-_NbCWgXzR1YJOF3LM7RGCW5N6_So7jdd_8OfS-2eo9IsoAxJjpcAS57Wf0ZWikrlqz_25efsjZgH" width="20%" height="20%">

One way to solve optimization problems is coordinate descent, in which at each step we optimize over one component of the unknown parameter vector, fixing all other components. The descent path so obtained is a sequence of steps, each of which is parallel to a coordinate axis in Rd, hence the name. It turns out that for the Lasso optimization problem, we can find a closed form solution for optimization over a single component fixing all other components.

In the Numpy code, we will implement the coordinated Descent Algorithm (taken from class slides "Lec 04 Regularization" page 23) :

<img src="https://lh5.googleusercontent.com/uO9HEP0VhUT9By-FmaPnHty2JD79nrPrDFydnwc__dQ9zOTgKGAc61X8abw-S37dLSssQzxy19JERIu2OGFpAlD4aMEAEBTiOq5DJd6J8bKunmX2CBwa3IkPkhqhusDUVTeFOQSK" width="40%" height="40%">

### 3.2 Summary of Lasso Regression’s Implementation

-   Dataset
    
In this extension, we decided to use the Boston Housing (Dataset used in HW4) and NYSE Stock market prices dataset (from outside class) [Data Source: [https://www.kaggle.com/dgawlik/nyse/kernels](https://www.kaggle.com/dgawlik/nyse/kernels)]

-   Sklearn implementation of Linear, Ridge and Lasso regressions  

Since in homework 4 we implemented the ridge regression, we think it’s interesting to see how pure linear regression, ridge regression and lasso regression works on the same dataset.

<img src="https://lh3.googleusercontent.com/lGAdAeN4aJLMuOWjdGIiBnvWhaPSuCKaMTsn81Dd32IWZg39_PdSZcE_lkUshpzkV_gkxyP1aFvHWTU4U7ftYpyFbmcV7rs92jttowa-FCLZMsaSQwxH8d3T05CJVfPA_wQh5zAD" width="60%" height="60%">

Remember that we did Ridge regression in class with k-fold validation and found the best alpha for the housing price dataset which is alpha = 1.49, so we will use this alpha when implementing Ridge Regression from Sklearn here.

<img src="https://lh4.googleusercontent.com/udMrbBJQivVS07p6DHIN4o19yUmpkS0ebqYebSzOF77J2mGq4crw-DrC-gCpLq66PYc0k0hI1VZp5NPhbP6FOefFnPGWKVSf5BCWwW2h6-3ddKZbgiVjT0OWg0q_3jCIygQEzIqh" width="60%" height="60%">
 
And for Lasso, we used alpha = 0.03 in Sklearn implementation.

<img src="https://lh5.googleusercontent.com/dKk3cRlR8JmGD57VZrhO7Sx5CW_O6aiqrN3mVZ1_XQkZlQe_4kU4nN__v2_3xiTz45iWJ99klv8FNZ3gcF-5rHLY_ZMPJbrVkHca_EvZLxCYRxcc2Wr2zY8LlrymhuQECYGHbqPh" width="60%" height="60%">

-   Numpy Implementation
    
We set alpha to 0.05 and the max number of iteration for coordinate descent for 1,000![](https://lh6.googleusercontent.com/31nwvTj1aPpVslHER4MHQ5JH5MDiw-7Dpy-_29MefkWSclWUNTHqAj_NAPgvCr0h9w-yoJco8aEtsischX9P7q71qRsM4n91taE-vj6EDHLlvPdlZYw9WsOxWBDuEvOuiwHfE8Xq)

<img src="https://lh6.googleusercontent.com/iYP-_rbx717OUZCCsA5IAXcSK8xa9kd096u6ZoKtyQRK-vGVGVElUa72wg9tdQ7_uvUyqeuH_ie7_L7Q5zcMF1pxkC0_vBsYPa1QTqGbxrEUv2dAGdppzOxBaYsK9wHgGjBZND-f" width="60%" height="60%">

### 3.3 Accuracies Table for Lasso

Note for regression problem, we used Mean Squared Error (MSE), Mean Absolute Error(MAE), R-square scores(R2) to measure accuracies:

<img src="https://lh6.googleusercontent.com/B7i_a5Vzl8KEoBKJj_ziawaDn-ko-XSb6XBijBUc_psyG9WQK8Dcm01sFhMYXAMYeBBIY0fAxWCuiGBbmoE8GJqgzNowxjzditTEJQf9N7kZHt8CHq-qc3vyOAyyhlEVFlJk970F" width="50%" height="50%">

One reason for why Lasso on GOOGLE Stock price's MSE is very high is because the y values which are the stock prices values are very large in themselves and the data has some outliers. Over all, since the R2 scores are pretty high, we can see that our models fits the dataset pretty well.


### 3.4 Explanations on Specific Datasets Lasso
Let’s see how well our predictions of y compared to the real label of y for each Dataset:
-   Boston housing:
   
![](https://lh6.googleusercontent.com/6VelUOGi_l2aUtLjMYVj-Oqqan41Pb8LHhuLHsNQ-7oDC5-1zHxP7GevtEx2MO_oJbEtv8yVyxlSA7U8aTywQuVTYpaVQczFGIA1E46UWdiXol-MKi_F908SKz0xT5S0j0Fn30fp)

We can see here both three regression from SKlearn has pretty similar performance, one reason might be we don’t have many outliers here in the Boston housing dataset and the Sklearn’s linear regression doesn’t overfit or under fit, it’s already very good.


![](https://lh3.googleusercontent.com/KoQGoc-4oOnNTDJJXu7U0x3nTZ9Wcea8qy8KOJsf3RKPpz0eIWzlAtHr-ctb-n5uK2sNR9pmoucVJFhhApYRzcoOwjAFTIrl_52YL1rDVsbXPk3AVosIhNQ7u_7DD2UXqTbjo1GG)

As we can see here and the accuracies table above, Lasso and Ridge perform better than Linear on NYSE stocks market both Sklearn implementation and our own Numpy implementation. This might because the original dataset has some terrible outliers so the Linear model might over-fit.

### 3.5 Numpy code for Lasso

![](https://lh3.googleusercontent.com/VrTExdNSKOXrUyh-H8x4a8b41dfZRIJwAmIgRlxpsXV2ueCOddolBvuE0VgoEToq74fwMksIY40ylWpQFmE2_EqhFs6bNgVLgIABvqVuyTNJMNvXJEVuIURIzD9zl-vjFev7LZVO)

![](https://lh6.googleusercontent.com/dwfAN_591sh3ez7nX4IDQlChgtvihz9p4Oyf160jrTt9qfaH79ynbLZ4-G9QO8fpQ7JQOtK0Ch52cxwRVynVlzlx0Vms8fNazMKHZjmiHq0pONECrxUM9d0nPmZU9u_tHPPxVg1e)

![](https://lh4.googleusercontent.com/eswgyxAJUxCBinNYnsdsIQH_prxDtp6q_kQhlBDL4gp2QIHQhcxHYAS-PK3kYhf0Sc-EGmVAzoJoHujAfvecB6PN3oAsDEOE_ig32YKDddStQqHscDSwr9Pyv2ORfJOjN2Rs5_cU)

  

### 3.6 Further explanation on Lasso regression

Although for most datasets that Lasso regression and Ridge Regression can achieve similar MSEs and R2 scores in terms of regularization.

There’s one feature that only Lasso regression has can be very useful -- feature selection.

This is because the coefficient of Lasso regression path can converge to 0 while it never happens in Ridge regression.

For example, the chart below showed the Lasso Path on Boston data:

![](https://lh3.googleusercontent.com/4HvF3f7c57JbeIMWHpDeAmudVwrDtnbvmqGd1I0iyRduu1ogd-taDIyzP4_-nmQAbz29s6Fp90gVEQh6TeGa1LKjw6FIdbP3vJXz0C7bLcsccjXq9Mh5zntMTyj-Zqy7XWYbwnia)

## 4 Neural Networks with Convolutional layers (CNN)

### 4.1 Explanation about CNN Algorithm
The key operation performed in CNN layers is that of 2D convolution. In homework8, we implemented neural networks with fully connected layers.

Difference between fully connected layers and convolutional layers:

In a fully connected layer, each neuron receives input from every element of the previous layer. In a convolutional layer, neurons receive input from only a restricted subarea of the previous layer.

In CNNs, each member of the kernel is used at every feasible position of the input. The parameter sharing used by the convolution operation means that rather than learning a separate set of parameters for  every location, we learn only one set.

![](https://lh4.googleusercontent.com/CNIaIrdI2Us_KtMWjlZwRZ3yIUHklIs9Ug6qmmxffwc5j7DqwxU8d0tCMbOV8IHJHfbJQx9oPg9n4SUobKaKtjNjtEoTgADdyjZJAU7F9kloILHfw-wz7KxaIDl8W_zmWGXV82W3)

### 4.2 Summary of CNN implementation

1.  2 Convolution layer which has several filters
    
2.  The weights are initialized randomly
    
3.  Using ReLU as activation function
    
4.  Using MaxPool Method when passing the layer
    
5.  Flatten the layer at the end with softmax
    
6.  When training, using gradient descent as well
    
7.  Using forward and backpropagation to update parameters

### 4.3 Accuracies Table 

![](https://lh3.googleusercontent.com/jD7lzyJi5zY3qiv4Zg2guiM4xUFtQz52kJrz11SdVp6ybvOVcmdNLMAjDsivSqJYNQZ3y-SfK1yn4liSHK7EShTkhocPkUR1Vh8b2Riy0fhSsLeQ7ChI8AZsSZ4UX83JhnCcf7v-)

### 4.4 Specific explanation on datasets

The dataset using is Minst handwritten digit database [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/), which contains images of handwritten digit from 0 -9.

Since it takes a long time to train, there’s only one final result. The accuracy of CNN varies a lot with different parameters. For example, the accuracy of Numpy implement would drop to 83.6% for the dataset in homework. There might be some difference from the gradient descent function of Pytorch and numpy implementation so that they didn’t have the same performance with the same hyper parameters. We try to find the best hyper parameters for both ones, so we choose different alpha as well as the params of layers. The reason why numpy implementation performs better that Pytorch is perhaps because of differences in implementation and Pytorch didn’t have optimal hyper parameters.

CNN performs better than NN because it uses the maxpool method to filter the most significant digit with a moving window, and it can extract the shape from the figure by convolution if there’s an obvious shape. This is useful to identify features from the figure.

### 4.5 Numpy Code 
See py notebook under CNN folder
