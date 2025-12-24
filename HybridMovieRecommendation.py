from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.preprocessing import normalize
from scipy import stats


main = tkinter.Tk()
main.title("Optimization of the Hybrid Movie Recommendation System Based on Weighted Classification and User Collaborative Filtering Algorithm") 
main.geometry("1300x1200")

global filename, cluster, X, Y
global hr, arhr, movies, users, ratings
global X_train, X_test, y_train, y_test, cs, kmeans_instance, metric

def upload(): 
    global filename, movies, users, ratings
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    #reading movies, ratings and user dataset 
    ratings = pd.read_csv('Dataset/ratings.csv', nrows=10000,sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
    movies = pd.read_csv('Dataset/movies.csv',encoding='latin-1',sep='\t', usecols=['movie_id', 'title', 'genres'])
    users = pd.read_csv('Dataset/users.csv', encoding='latin-1',sep='\t')
    #replacing missing values with 0
    ratings.fillna(0, inplace = True)
    text.insert(END, "User Details \n\n"+str(users.head())+"\n\n")
    text.insert(END, "Movie Details \n\n"+str(movies.head())+"\n\n")
    text.insert(END, "Rating Details \n\n"+str(ratings.head())+"\n\n")
    
def sparseModel():
    global ratings, X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    ratings.fillna(0, inplace = True)
    dataset = ratings.values
    #converting ratings in to sparse matrix
    matrix = ratings.values
    #extracting user and movie id as X features and RATINGS as Y and this features are called as sparse matrix
    X = matrix[:,0:2]
    Y = matrix[:,2]
    text.insert(END,"Sparse Matrix Extracted from Dataset\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")

def pearson_dist(x, y): #function to calculate similarity between X user and Y user 
    x = np.asarray(x)
    y = np.asarray(y)
    a_norm = np.linalg.norm(x)
    b_norm = np.linalg.norm(y)
    stats.pearsonr(x, y)[0]
    similiarity = np.dot(x, y.T)/(a_norm * b_norm)
    similiarity = 1. - similiarity
    return similiarity #return pearson similarity

def getNearestNeighbors(test): #function to calculate nearest neighbour using weighted threshold
    list_of_test_users = np.zeros((len(test), 5))
    for index_point in range(len(test)): #loop all test users and then find users similarity
        list_of_test_users[index_point] = [metric(test[index_point], c) for c in cs]
    label = np.argmax(list_of_test_users, axis=1) #extract maximum similarity user as label
    return label #return maximum similarity user value

def runLocalClustering():
    global ratings, X, Y, cs, hr, arhr, metric, kmeans_instance
    global X_train, X_test, y_train, y_test
    hr = []
    arhr = []
    text.delete('1.0', END)
    metric = distance_metric(type_metric.USER_DEFINED, func=pearson_dist) #here we are creating distance similarity measure with pearson function
    #defining initial center points
    initial_centers = [[0.023917746175045114, 0.9995049723305172], [0.8574672718847574, 0.49772130656070934],
                       [0.5351297755937158, 0.8354536941067743], [0.988826874252877, 0.12248444970535248], [0.18061215808615247, 0.9812169115929825]]
    kmeans_instance = kmeans(X_train, initial_centers, metric=metric, tolerance = 0.01, ccore = False)#creating KMEANS with given X train data, initial center and metric
    kmeans_instance.process() #start grouping similar behaviour user into same cluster
    clusters = kmeans_instance.get_clusters() #return number of clusters
    cs = initial_centers
    label = kmeans_instance.predict(X_train)#local hit rate will be calculated by using local KMEANS clustering object
    for i in range(len(label)):
        label[i] = label[i] + 1
    hitrate = 0
    for i in range(len(label)):
        if label[i] == y_train[i]:
            hitrate += 1
    total_test_users = len(X_test) / 2
    hitrate = (hitrate / total_test_users)
    hr.append(hitrate)
    avg = hitrate / 100
    arhr.append(avg)
    text.insert(END,"Local KMEANS Clustering Computation Completed\n\n")
    text.insert(END,"Propose TopPop Algorithm Local Hit Rate: "+str(hitrate)+"\n")
    text.insert(END,"Propose TopPop Algorithm Local Avg Hit Rate: "+str(avg)+"\n\n")

    label = getNearestNeighbors(X_test) #global hit rate calculation using maximum similarity by applying nearest neighbour formula with max weight
    for i in range(len(label)):
        label[i] = label[i] + 1
    hitrate = 0
    for i in range(len(label)):
        if label[i] == y_test[i]:
            hitrate += 1
    hitrate = hitrate / len(X_test)
    hr.append(hitrate)
    avg = hitrate / 100
    arhr.append(avg)
    
    text.insert(END,"Propose TopPop Algorithm Global Hit Rate: "+str(hitrate)+"\n")
    text.insert(END,"Propose TopPop Algorithm Global Avg Hit Rate: "+str(avg)+"\n")

def graph():
    global hr, arhr
    height = [hr[0], arhr[0], hr[1], arhr[1]]
    bars = ('Local HR','Local ARHR', 'Global HR', 'Global ARHR')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("HR & ARHR Comparison Graph")
    plt.show()

def close():
    main.destroy()


def predict():
    global kmeans_instance, X, Y, movies, ratings
    text.delete('1.0', END)
    user_id = int(text1.get("1.0",END))
    movie_id = int(text2.get("1.0",END))
    test = []
    test.append([user_id, movie_id])
    test = np.asarray(test)
    test = normalize(test)
    predict = kmeans_instance.predict(test)[0]
    predict = predict + 1
    print(predict)
    recommendationList = []
    movieList = movies.values
    ratingList = ratings.values
    for i in range(len(Y)):
        if Y[i] == predict:
            movieID = ratingList[i,1]
            for k in range(len(movieList)):
                if movieList[k,0] == movieID:
                    recommendationList.append(movieList[k,1])
        if len(recommendationList) >= 10:
            break
    if len(recommendationList) > 0:
        text.insert(END,"Predicted Rating Score is : "+str(predict)+"\n\n")
        text.insert(END,"Below are the Recommended Movie List\n\n")
        for k in range(len(recommendationList)):
            text.insert(END,recommendationList[k]+"\n")
    else:
        text.insert(END,"Unable to find recommendation for given user")


font = ('times', 16, 'bold')
title = Label(main, text='Optimization of the Hybrid Movie Recommendation System Based on Weighted Classification and User Collaborative Filtering Algorithm')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Movielens Dataset", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Calculate Sparse Linear Model", command=sparseModel, bg='#ffb3fe')
processButton.place(x=310,y=550)
processButton.config(font=font1) 

knnButton = Button(main, text="Run Local Clustering Algorithm", command=runLocalClustering, bg='#ffb3fe')
knnButton.place(x=600,y=550)
knnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=900,y=550)
graphButton.config(font=font1)

l1 = Label(main, text='User ID:')
l1.config(font=font1)
l1.place(x=50,y=600)

text1=Text(main,height=2,width=20)
scroll1=Scrollbar(text1)
text1.configure(yscrollcommand=scroll1.set)
text1.place(x=160,y=600)
text1.config(font=font1)

l2 = Label(main, text='Movie ID:')
l2.config(font=font1)
l2.place(x=370,y=600)

text2=Text(main,height=2,width=20)
scroll2=Scrollbar(text2)
text2.configure(yscrollcommand=scroll2.set)
text2.place(x=500,y=600)
text2.config(font=font1)

predictButton = Button(main, text="Run Weighted Classification", command=predict, bg='#ffb3fe')
predictButton.place(x=750,y=600)
predictButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
exitButton.place(x=50,y=650)
exitButton.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
