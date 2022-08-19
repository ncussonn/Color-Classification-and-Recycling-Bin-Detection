# Defines the Trainer Function for Pixel Classification

# For creating weights quickly
# w_r = Trainer(X,y,'r'); w_g = Trainer(X,y,'g'); w_b = Trainer(X,y,'b');

def Trainer(X,y,color,a,k):

    import numpy as np

    # y must be changed to an immuatble object to avoid altering for other trainings
    y = tuple(y)
    y = np.asarray(y)
    y = np.reshape(y,(len(y),1))

    def s(z):
		# Limit z to avoid overflow
        z = np.where(z >50,50,z)
        sigmoid = np.divide(1,(1+np.exp(-z)));
        return sigmoid

    def VectAugment(K):
        for i in range(0,len(y)):
            if y[i] != K:
                y[i] = -1
            else:
                y[i] = 1

    # Augmenting y matrix to work for 3 different Binary Logistic Regressions
    # Red
    if color == 'r':
        VectAugment(1)
    # Green
    elif color == 'g':
        VectAugment(2)
    # Blue
    elif color == 'b':
        VectAugment(3)
    else:
        print("Invalild color - please input r,g,b")
        return

    omega = np.zeros((len(X[0]),1), dtype=float)

    yX = np.multiply(y,X);

    # Gradient Descent - k amount of times:
    for j in range(0,k):

        sigmoid = s(yX @ omega)
        Sum = np.array([sum(np.multiply(yX,(1-sigmoid)))])
        omega = omega + a*np.transpose(Sum)

    print("Final Omega")
    print(omega)

    return omega