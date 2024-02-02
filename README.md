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
Neural Network of FFM architecture to solve XOR problem. This is a hello-world project to practise mathematical apparatus behind deep learning.

# 2. Problem
XOR function via FFM architechture with no use of frameworks.

# 3. Method
Algorythm structure is comprised of 5 parts: Dataset, Model, Cost function, Optimization procedure and Initializing data. Describtion of every part is provided below.
## 3.1. Dataset
$$𝕏=\{[0,0]^T, [0,1]^T, [1,0]^T, [1,1]^T\}$$
## 3.2. Model
### > FFM architecture:
![alt text](<link to graph>)
$$
\text{Fig.1: Architecture topology}
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
&b-\text{biases vector}~f^{(2)} \\
&h-\text{output vector}~f^{(1)} \\
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
&g(z)=max\{0, z\}-\text{activation function ReLU}
\end{aligned}
$$

The input function $z$ is a default affine transformation allowing learning algorythm to manipulate data to find representation that reduces error.

### > Composition
$$
f(x; W,c,w,b) = f^{(2)}( f^{(1)}( x)) = w^Tmax\{0, W^Tx+c\}+b
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
&f^{*}-\text{true distribution function} \\
&f-\text{approximating distribution function}
\end{aligned}
$$
### > Optimization procedure
$$
<N/A>
$$
In particular, the problem doesn't need optimization due to guaranteed convergence of linear equation of linear regression model.\
In general, the stochastic gradient descend is usually used which gives acceptable results.

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
\<wip\>

# 5. Results
\<wip\>