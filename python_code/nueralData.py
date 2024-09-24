# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset where each row is a training example
X = np.empty((200,2))
for i in range(200):
    X[i] = pca_data[i]
    #print(X[i])
print(X.shape)

# output dataset where each row is a training example
y = np.empty((200,1))
for i in range(200):
    y[i] = mean_numbers[i+1][0]
    print(y[i])
