# Contents:
1. Describtion
2. Problem
3. Method
    * 3.1. Dataset
    * 3.2. Model
    * 3.3. Cost function
    * 3.4. Optimization procedure
    * 3.5. Initializing data
4. Code
5. Results

# 1. Describtion
One of the most important steps in development of Deep Learning as a field was solving the XOR problem by moving away from binary step as an activation function. First, switched to sigmoid functions and later to ReLU (rectified linear unit), Neural Networks acquired the ability to approximate to any function (see 'Universal Approximation Theorem', Hornik et al., 1989; Cybenko, 1989).

One of the first universal network architectures to solve general approximation problems was the FNN (Feedforward Neural Network) architecture that's widely used to this day.

This project presents the mathematical apparatus of the FNN architecture to solve XOR problem and the implementing python code.

IMPORTANT: This project doesn't include the optimization procedure. To see how the optimization by Stochastic Gradient Descent works, check my another project by following link: https://github.com/AKAD0/FNN.SGD_MNIST.git

# 2. Problem
XOR function via FNN architechture with no use of frameworks.

# 3. Method
Algorithm structure is comprised of 5 parts: Dataset, Model, Cost function, Optimization procedure and Initializing data. Describtion of every part is provided below.
## 3.1. Dataset
$$
𝕏=\lbrace{[0,0]^T, [0,1]^T, [1,0]^T, [1,1]^T\rbrace},~
f^*=\begin{bmatrix}
    0 \\
    1 \\
    1 \\
    0 \\
    \end{bmatrix}
$$
## 3.2. Model
### > FNN architecture:
<p align="center">
  <img src="https://github.com/AKAD0/FNN_XOR/blob/master/Fig1.png">
</p>

$$
\text{Fig.1: Topology of the architecture}
$$
### > Output layer:
$$
y=f^{(2)}(h; w,b) = w^Th+b
$$

$$
\\
\begin{aligned}
\text{where}~
&w-\text{weights vector of}~f^{(2)} \\
&b-\text{biases vector of}~f^{(2)} \\
&h-\text{output vector of}~f^{(1)} \\
\end{aligned}
$$

The function $f^{(2)}$ is linear because $h=f^{(1)}$ returns data solvable for linear functions. 

### > Hidden layer:
$$
h = f^{(1)} = g( z( x; W,c))
$$

$$
\\
\begin{aligned}
\text{where}~
&x-\text{input vector} \\
&W-\text{weights vector of}~f^{(1)} \\
&c-\text{biases vector of}~f^{(1)} \\
&z(x; W,c)=x^TW+c-\text{input function} \\
&g(z_i)=max\lbrace0, z_{:,i}\rbrace-\text{activation function ReLU}
\end{aligned}
$$

The input function $z$ is a default affine transformation allowing learning algorithm to manipulate data to find representation that reduces error. \
Activation function is applied to every element in the vector.

### > Input layer:
$$
X \in ℕ^{4×2}
$$

Every row represents a sample of two numbers to pass to XOR function.

### > Composition
$$
f(x; W,c,w,b) = f^{(2)}( f^{(1)}( x)) = w^Tmax\lbrace0, W^Tx+c\rbrace+b
$$
### > Cost Function
$$
J(θ) = \frac{1}{4}\sum_{x\in𝕏^4}( f^{*}(x) - f(x;θ))^2
$$

$$
\\
\begin{aligned}
\text{where}~
&θ-\text{optimizing parameters W,c,w,b} \\
&x-\text{input data} \\
&f^{*}-\text{true function} \\
&f-\text{approximating function}
\end{aligned}
$$
### > Optimization procedure
$$
\text{N/A}
$$

For this particular example optimal parameters are already known and provided as initializing data.\
In general, the stochastic gradient descent is usually used to find these parameters and produces acceptable results.

### > Initializing data
$$
W = \begin{bmatrix}
    1 & 1 \\
    1 & 1 \\
    \end{bmatrix}
,~
c = \begin{bmatrix}
    0 \\
    -1 \\
    \end{bmatrix}
,~
w = \begin{bmatrix}
    1 \\
    -2 \\
    \end{bmatrix}
,~
b=0
$$

# 4. Code
<p align="center">
  <img src="https://github.com/AKAD0/FNN_XOR/blob/master/Block.png">
</p>

$$
\text{Fig.2: Block scheme of the algorithm}
$$

The listing of the program is provided in the '/FNN_XOR.ipynb' file.

# 5. Results
MSE metric evaluated as '0.0' and simple human check of outputted results show that the algorithm solves the problem.