import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn 
from sklearn.decomposition import PCA

#imports the data 
import csv
file = open('SPYdata.csv')
type = file
csvreader = csv.reader(file)
header = []
header = next(csvreader)
header

rows = []
for row in csvreader:
    rows.append(row)


file.close()
import csv
file = open('VIXdata.csv')
type = file
csvreader = csv.reader(file)
vheader = []
vheader = next(csvreader)
vheader

vrows = []
for row in csvreader:
    vrows.append(row)



numbers = []


for i in range(len(rows)):
    for j in range(len(vrows)):
        if rows[i][0] == vrows[j][0]:
            rows[i].append(vrows[j][1])
        
            


    
    

    
#Puts the data into a new array in order to manipulate
for item in rows:
    numbers.append(item)


#takes out the date leaving with closing price, volume, and vix
#then converts this array to a floating point array
for i in range(len(rows)):
    numbers[i].pop(0)
    j = 0
    while j < 3:
        numbers[i][j] = float(numbers[i][j])
        j+=1
    
#changes to a numpy array then plots it
numbers = np.array(numbers)
#pyplot.scatter(numbers[:,0],numbers[:,1])
#pyplot.grid(True)
#pyplot.show()

#centers the data at 0
mean = np.mean(numbers, axis= 0)
mean_numbers = numbers - mean
print("Mean ", mean)
#pyplot.scatter(mean_numbers[:,0],mean_numbers[:,1], mean_numbers[:,2])
#pyplot.grid(True)
#pyplot.show()

#gets covariance matrix of the data
cov = np.cov(mean_numbers.T)
cov = np.round(cov, 2)
print(cov)

#gets eigen values of covariance matrix
eig_val, eig_vec = np.linalg.eig(cov)
print("Eigen vectors ", eig_vec)
print("Eigen values ", eig_val, "\n")

#sorts the eigen vecotrs
indices = np.arange(0,len(eig_val), 1)
indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
eig_val = eig_val[indices]
eig_vec = eig_vec[:,indices]
print("Sorted Eigen vectors ", eig_vec)
print("Sorted Eigen values ", eig_val, "\n")

#gets the cumulative and explained variance
sum_eig = np.sum(eig_val)
exp_variance = eig_val/sum_eig
cum_variance = np.cumsum(eig_val)
print(exp_variance)
print(cum_variance)

#plots the eigen vectors in 3d plot
#ax = pyplot.figure().add_subplot(projection='3d')
#quiv = ax.Axes3D.quiver()
# Creating plot
#ax.scatter3D(mean_numbers[:,0], mean_numbers[:,1], mean_numbers[:,2], color = "green")
#ax.quiver(0,0, 0,eig_vec[0,0], eig_vec[1,0], eig_vec[2,0], color=['r'], length=200)
#ax.quiver(0,0, 0,-eig_vec[0,0], -eig_vec[1,0], -eig_vec[2,0],color=['r'], length=200)
#ax.quiver(0,0, 0,eig_vec[0,1], eig_vec[1,1], eig_vec[2,1], color=['y'], length=200)
#ax.quiver(0,0, 0,-eig_vec[0,1], -eig_vec[1,1], -eig_vec[2,1], color=['y'], length=200)
#ax.quiver(0,0,0, eig_vec[0,2], eig_vec[1,2], eig_vec[2,2], color = ['b'], length = 200)
#ax.quiver(0,0,0, -eig_vec[0,2], -eig_vec[1,2], -eig_vec[2,2], color = ['b'], length = 200)
#pyplot.show()

#finds the principle components and graphs the 2 most significant
pca_data = np.dot(mean_numbers, eig_vec)
print("Transformed data ", pca_data.shape)
pyplot.plot(pca_data[:,0],pca_data[:,1],'.')
pyplot.show()

#transfroms the data from 3 parameters to 2, dropping the principle component of least significance, then graphs it (not needed to do, just a quicker way)
#produces same result as lines above
pca = PCA(n_components = 2)
pca.fit(mean_numbers)
pca_data = pca.transform(mean_numbers)
print(pca_data.shape)
print(mean_numbers.shape)
pyplot.plot(pca_data[:,0],pca_data[:,1],'.')
pyplot.xlabel('First Principal Component')
pyplot.ylabel('Second Principal Component')
pyplot.show()







