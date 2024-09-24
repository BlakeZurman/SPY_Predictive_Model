import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn

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


