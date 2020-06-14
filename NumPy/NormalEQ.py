import math
import numpy as np


def cholesky_factorize(A):
    """Return the Cholesky Factorization L of A, where
        * A is an nxn symmetric, positive definite matrix
        * L is lower triangular, with positive diagonal entries
        * $A = LL^T$
        
    >>> M = np.array([[8., 3., 2.],
                      [3., 5., 1.],
                      [2., 1., 3.]])
    >>> L = cholesky_factorize(M)
    >>> np.matmul(L, L.T)
    array([[8., 3., 2.],
           [3., 5., 1.],
           [2., 1., 3.]])
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for k in range(i+1):
            s = sum(L[i,j] * L[k,j] for j in range(k))
            if(i==k):
                L[i,k] = math.sqrt(A[i,i] - s)
            else:
                L[i,k] = 1.0 / L[k,k] * (A[i,k] - s)
    return L

def forward_substitution(A, b):
    """Return a vector x with np.matmul(A, x) == b, where 
        * A is an nxn numpy matrix that is lower-triangular and non-singular
        * b is a nx1 numpy vector
        
    >>> A = np.array([[2., 0.],
                      [1., -2.]])
    >>> b = np.array([1., 2.])
    >>> forward_substitution(A, b)
    array([ 0.5 , -0.75])
    """
    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float)
    for i in range(0, n, 1):
        s = 0
        for j in range(0, i, 1):
            s += A[i,j] * x[j]
        x[i] = (b[i] - s) / A[i,i]
    return x


def solve_normal_equation(A, b):
    """
    Return the solution x to the linear least squares problem
        $$Ax \approx b$$ using normal equations,
    where A is an (m x n) matrix, with m > n, rank(A) = n, and
          b is a vector of size (m)
    """
    sym = np.matmul(A.T, A)
    lower = cholesky_factorize(sym)
    newB = np.matmul(A.T, b)
    newLower = np.matmul(lower, lower.T)
    return forward_substitution(newLower, newB)
    
    
irisA = np.array([[4.8, 6.3, 5.7, 5.1, 7.7, 5.6, 4.9, 4.4, 6.4, 6.2, 6.7, 4.5, 6.3,
                   4.8, 5.8, 4.7, 5.4, 7.4, 6.4, 6.3, 5.1, 5.7, 6.5, 5.5, 7.2, 6.9,
                   6.8, 6. , 5. , 5.4, 5.6, 6.1, 5.9, 5.6, 6. , 6. , 4.4, 6.9, 6.7,
                   5.1, 6. , 6.3, 4.6, 6.7, 5. , 6.7, 5.8, 5.1, 5.2, 6.1],
                  [3.1, 2.5, 3. , 3.7, 3.8, 3. , 3.1, 2.9, 2.9, 2.9, 3.1, 2.3, 2.5,
                   3.4, 2.7, 3.2, 3.7, 2.8, 2.8, 3.3, 2.5, 4.4, 3. , 2.6, 3.2, 3.2,
                   3. , 2.9, 3.6, 3.9, 3. , 2.6, 3.2, 2.8, 2.7, 2.2, 3.2, 3.1, 3.1,
                   3.8, 2.2, 2.7, 3.1, 3. , 3.5, 2.5, 4. , 3.5, 3.4, 3. ],
                  [1.6, 4.9, 4.2, 1.5, 6.7, 4.5, 1.5, 1.4, 4.3, 4.3, 5.6, 1.3, 5. ,
                   1.6, 5.1, 1.3, 1.5, 6.1, 5.6, 6. , 3. , 1.5, 5.8, 4.4, 6. , 5.7,
                   5.5, 4.5, 1.4, 1.3, 4.1, 5.6, 4.8, 4.9, 5.1, 5. , 1.3, 4.9, 4.4,
                   1.5, 4. , 4.9, 1.5, 5.2, 1.3, 5.8, 1.2, 1.4, 1.4, 4.9]])
irisb = np.array([0.2, 1.5, 1.2, 0.4, 2.2, 1.5, 0.1, 0.2, 1.3, 1.3, 2.4, 0.3, 1.9,
                  0.2, 1.9, 0.2, 0.2, 1.9, 2.1, 2.5, 1.1, 0.4, 2.2, 1.2, 1.8, 2.3,
                  2.1, 1.5, 0.2, 0.4, 1.3, 1.4, 1.8, 2. , 1.6, 1.5, 0.2, 1.5, 1.4,
                  0.3, 1. , 1.8, 0.2, 2.3, 0.3, 1.8, 0.2, 0.2, 0.2, 1.8])
A = irisA.T # transpose
b = irisb

iris_x = solve_normal_equation(A, b)
iris_residual = np.linalg.norm(np.matmul(A, iris_x) - b)


def householder_v(a):
    """Return the vector $v$ that defines the Householder Transform
         H = I - 2 np.matmul(v, v.T) / np.matmul(v.T, v)
    that will eliminate all but the first element of the 
    input vector a. Choose the $v$ that does not result in
    cancellation.

    Do not modify the vector `a`.
    
    Example:
        >>> a = np.array([2., 1., 2.])
        >>> householder_v(a)
        array([5., 1., 2.])
        >>> a
        array([2., 1., 2.])
    """
    v = np.copy(a)
    sign = 1
    if(np.sign(v[0]) == -1):
        sign = -1
        
    norm = np.linalg.norm(a) * sign
    v[0] = v[0] + norm
    return v
    


def apply_householder(v, u):
    """Return the result of the Householder transformation defined
    by the vector $v$ applied to the vector $u$. You should not
    compute the Householder matrix H directly.
    
    Example:
    
    >>> apply_householder(np.array([5., 1., 2.]), np.array([2., 1., 2.]))
    array([-3.,  0.,  0.])
    >>> apply_householder(np.array([5., 1., 2.]), np.array([2., 3., 4.]))
    array([-5. ,  1.6,  1.2])
    """
    return u - 2 * (np.matmul(v.T, u) / np.matmul(v.T, v)) * v
    

def apply_householder_matrix(v, U):
    """Return the result of the Householder transformation defined
    by the vector $v$ applied to all the vectors in the matrix U.
    You should not compute the Householder matrix H directly.
    
    Example:
    
    >>> v = np.array([5., 1., 2.])
    >>> U = np.array([[2., 2.],
                      [1., 3.], 
                      [2., 4.]])
    >>> apply_householder_matrix(v, U)
    array([[-3. , -5. ],
           [ 0. ,  1.6],
           [ 0. ,  1.2]])
    """
    p = 2 / np.matmul(v, v.T)
    pv = p * v
    vtu = np.matmul(v.T, U)
    out = np.outer(pv, vtu)
    return U - out



def qr_householder(A, b):
    """Return the matrix [R O]^T, and vector [c1 c2]^T equivalent
    to the system $Ax \approx b$. This algorithm is similar to
    Algorithm 3.1 in the textbook.
    """
    for k in range(A.shape[1]):
        v = householder_v(A[k:, k])
        if np.linalg.norm(v) != 0:
            A[k:, k:] = apply_householder_matrix(v, A[k:, k:])
            b[k:] = apply_householder(v, b[k:])
    # now, A is upper-triangular
    return A, b

def solve_qr_householder(A, b):
    """
    Return the solution x to the linear least squares problem
        $$Ax \approx b$$ using Householder QR decomposition.
    Where A is an (m x n) matrix, with m > n, rank(A) = n, and
          b is a vector of size (m)
          0 matrix in [R 0] is (n - m) x m
    """
    A, b = qr_householder(A, b)
    shape = np.shape(A)
    m = shape[0] 
    n = shape[1] 
    zeroSize = n - m
    A = A[:zeroSize, :]
    b = b[:zeroSize]
    return np.linalg.solve(A, b)


# The code below loads the data, splits it into "train" and "test" sets,
# and plots the a subset of the data. We will use `train_images` and `train_labels`
# to set up Linear Least Squares problems. We will use `test_images` and `test_labels`
# to test the models that we build.

mnist_images = np.load("mnist_images.npy")
test_images = mnist_images[4500:]  #  500 images
train_images = mnist_images[:4500] # 4500 images
mnist_labels = np.load("mnist_labels.npy")
test_labels = mnist_labels[4500:]
train_labels = mnist_labels[:4500]

def plot_mnist(remove_border=False):
    """ Plot the first 40 data points in the MNIST train_images. """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for i in range(4 * 10):
        plt.subplot(4, 10, i+1)
        if remove_border:
            plt.imshow(train_images[i,4:24,4:24])
        else:
            plt.imshow(train_images[i])
            plot_mnist() # please comment out this line before submission

# The number of examples of each number in the dataset
mnist_digits = [0,0,0,0,0,0,0,0,0,0]
for x in train_labels:
    mnist_digits[x] += 1
    
#
# We will build a rudimentary model to predict whether a digit is the
# digit `0`. Our features will be the intensity at each pixel.
# There are 28 * 28 = 784 pixels in each image. However, in order
# to obtain a matrix $A$ that is of full rank, we will ignore the
# pixels along the border. That is, we will only use 400 pixels
# in the center of the image.
# Plot the MNIST images, with only the center pixels that we will use
# in our digit classification model.
plot_mnist(True)


# We will now build a rudimentary model to predict whether a digit is the
# digit `0`. To obtain a matrix of full rank, we will use the 400 pixels
# at the center of each image as features. Our target will
# be whether our digit is `0`.
# 
# In short, the model we will build looks like this:
# 
# $$x_1 p_1 + x_2 p_2 + ... + x_{400} p_{400} = y$$
# 
# Where $p_i$ is the pixel intensity at pixel $i$ (the ordering of the pixel's
# doesn't actually matter), and the value of $y$ determines whether our digit is a `0` or not.
# 
# We will solve for the coefficients $x_i$ by solving the linear
# least squares problem $Ax \approx b$, where $A$ is constructed using
# the pixel intensities of the images in `train_images`, and $y$ is constructed
# using the labels for those images. For convenience, we will set $y = 1$ for
# images of the digit `0`, and $y=0$ for other digits.

A = train_images[:, 4:24, 4:24].reshape([-1, 20*20])
b = (train_labels == 0).astype(np.float32)

mnist_m = 4500
mnist_n = 400


mnist_x = solve_qr_householder(A, b)
mnist_r = np.linalg.norm(np.matmul(A, mnist_x) - b)


import matplotlib.pyplot as plt 
plt.imshow(test_images[0])
p = test_images[0, 4:24, 4:24].reshape([-1, 20*20])
test_image_0 = True
test_image_0_y = np.dot(p[0], mnist_x)

reshapeTestImages = test_images[:, 4:24, 4:24].reshape([-1, 20*20])
test_image_y = np.dot(reshapeTestImages, mnist_x)



test_image_pred = np.copy(test_image_y)
l = np.shape(test_image_pred)[0]
for i in range(l):
    if test_image_pred[i] >= 0.5:
        test_image_pred[i] = 1
    else:
        test_image_pred[i] = 0
test_accuracy = sum(test_image_pred == (test_labels == 0).astype(float)) / len(test_labels)
print(test_accuracy)


def mnist_classifiers():
    """Return the coefficients for linear least squares models for every digit.
    
    Example:
        >>> xs = mnist_classifiers()
        >>> np.all(xs[0] == mnist_x1)
        True
    """
    # you can use train_images and train_labels here, and make
    # whatever edits to this function as you wish.
    A = train_images[:, 4:24, 4:24].reshape([-1, 20*20])  
    xs = []
    for x in range(0, 10):
        b = (train_labels == x).astype(np.float32)
        mnist = solve_normal_equation(A, b)
        xs.append(mnist)        
    return np.stack(xs)

def prediction_accuracy(xs):
    """Return the prediction 
    """
    testA = test_images[:, 4:24, 4:24].reshape([-1, 20*20])
    ys = np.matmul(testA, xs.T)
    pred = np.argmax(ys, axis=1)
    return sum(pred == test_labels) / len(test_labels)

