#Name:Mickey Chen
#Email:Mickey.chen90@myhunter.cuny.edu

#resources:https://realpython.com/python-gui-tkinter/
#https://docs.python.org/3/library/tkinter.html
#these are for k-nearest-neighbor
#https://www.analyticsvidhya.com/blog/2020/08/recommendation-system-k-nearest-neighbors/
#https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea
#https://www.youtube.com/watch?v=kccT0FVK6OY
#these are for data source
#https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset?select=IMDb+movies.csv
#https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset?select=IMDb+ratings.csv
#this is for data visualization
#https://www.dataquest.io/blog/making-538-plots/
#this is for numpy squeeze
#https://www.pythonpool.com/numpy-squeeze/
#this is outside human resource 
#Yu-Zhu (My cousin, a college student(not a hunter college student) who is also studying computer science 
#helped me with some part of K-nearest neighbor and Tkinter to make the recommendation system)

#Title:The Movie Viewer List
#URL:https://q915925914.wixsite.com/movie-view

#importing libraries
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
from tkinter import ttk

def movie_recommender(movie_name,df,movie_df,knn,data): 
    n_movies_to_reccomend = 10 #selecting how many movies need to be selected
    movie_list = movie_df[movie_df['title'].str.contains(movie_name,case = False)] #Search the movie in Movie_ID_File 
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId'] # index of searched movie
        movie_idx = df[df['movieId'] == movie_idx].index[0]
        # using KNN algorithm assumes that similar things exist in close proximity
        Recommendation , indices = knn.kneighbors(data[movie_idx],n_neighbors=n_movies_to_reccomend+1) # K-Nearest-Neighbor Machine Learning model that was trained on the file.csv file  
        #sort recommendation into a list
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),Recommendation.squeeze().tolist())),key=lambda x: x[1])[:0:-1] # index of recommended movies
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = df.iloc[val[0]]['movieId']
            idx = movie_df[movie_df['movieId'] == movie_idx].index # finding movie  names in Movie_ID_File
            recommend_frame.append({'Title':movie_df.iloc[idx]['title'].values[0],'Recommendation':val[1]})# compiling result  
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No Movie Found. Check your Input Please"

#later on when the recommendation system is finish this will be the format of the layout 
def display_recommendation(result_df):
    #make this example reproducible
    np.random.seed(0)

    #define figure and axes
    fig, ax = plt.subplots()

    #hide the axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    #create data
    df = pd.DataFrame(result_df, columns=['Title', 'Recommendation'])

    #create table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    #display table
    fig.tight_layout()
    plt.show()

#select the Movie_Id_Titles.csv File
#movie_df = pd.read_csv(io.StringIO(uploaded['Movie_Id_Titles.csv'].decode('ISO-8859-1'))) # reading csv file
movie_df = pd.read_csv('Movie_Id_Titles.csv',encoding = "ISO-8859-1") # reading csv file 

#select the File.tsv File
#rating_df = pd.read_csv(io.StringIO(uploaded['file.tsv'].decode('utf-8')), sep='\t') and convert plain string to unicode 

rating_df = pd.read_csv('file.tsv', sep='\t')

#rating_df.column = ['userId','movieId','rating','timestamp']


#adding coulumn name
rating_df.rename(columns={'0':'userId',
                          '50':'movieId',
                          '5':'rating',
                           '881250949':'timestamp'}, 
                  
                 inplace=True)

df = rating_df.pivot(index='movieId',columns='userId',values='rating') #making dataset usable
df.fillna(0,inplace=True) # replacing NaN with 0

no_user_voted = rating_df.groupby('movieId')['rating'].agg('count')

no_movies_voted = rating_df.groupby('userId')['rating'].agg('count')

#making the visualization in histogram of user voting by movie id 
f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index,no_user_voted,color='blue')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

df = df.loc[no_user_voted[no_user_voted > 10].index,:]

#making the visualization in histogram of user voting by user id 
f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_movies_voted.index,no_movies_voted,color='blue')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()

df=df.loc[:,no_movies_voted[no_movies_voted > 50].index]

data = csr_matrix(df.values) # removing sparsity 
df.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1) # configuring KNN Model to train

knn.fit(data) # Final Dataset Training

#enter name of movie which is present in dataset 
#movie_name = input("Enter Movie Name: ")


win= Tk()
win.title("Movie Recommendation System")
#Set the geometry of tkinter frame
win.geometry("750x250")
# get_value() is function for getting input value and display the results
def get_value():
    #retrieving text with .get
    e_text=entry.get()
    #set label background 
    Label(win, text=e_text, font= ('Century 15 bold')).pack(pady=20)
    result_df=movie_recommender(e_text,df,movie_df,knn,data)
    display_recommendation(result_df)

   
#Create an Entry Widget
entry= ttk.Entry(win,font=('Century 12'),width=40)
entry.pack(pady= 30)
#Create a button to display the text of entry widget
button= ttk.Button(win, text="Search", command= get_value)
button.pack()
# tells Python to run the Tkinter event loop. This method listens for events, such as button clicks or keypresses, 
#and blocks any code that comes after it from running until the window it’s called on is closed
win.mainloop()

