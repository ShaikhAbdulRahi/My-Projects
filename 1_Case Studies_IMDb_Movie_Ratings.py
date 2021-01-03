#!/usr/bin/env python
# coding: utf-8

# # 1.1: Read the Movies Data

# In[1]:


import pandas as pd


# In[2]:


movies=pd.read_csv("E:/Decode_Lectures/Case Study/Case Study_01/Movie+Assignment+Data.csv")
movies


# # 1.2: Inspect the Dataframe

# In[3]:


movies.isna().sum() # Inspect the null-values in the data frame


# In[4]:


movies.shape# Inspect the dimensions of the data frame


# In[5]:


movies.head()# Inspect the summary  of the data frame


# In[6]:


movies.info()# Inspect the information about the data frame


# In[7]:


movies.describe()# Inspect the information about numeric column of the data frame


# # Task 2: Data Analysis

# # 2.1: Reduce those Digits from "Budget " &"Gross "

# In[8]:


movies["Gross"]=movies["Gross"]/1000000
movies["budget"]=movies["budget"]/1000000
movies


# # 2.2: Let's Talk Profit!

# In[9]:


# Create the new column named 'profit' by ['budget' -'gross'] column
movies["profit"]=movies["Gross"]- movies["budget"]
movies


# In[10]:


#2Sort the dataframe using the profit column as reference
movies=movies.sort_values(by="profit")
movies


# In[11]:


#3.	Extract the top ten profiting movies in descending order and store them in a new dataframe - top10
movies=movies.sort_values(by="profit",ascending=False)
movies
movies.iloc[:10,:] # all columns of top 10 rows


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.jointplot("budget","profit",movies)
plt.show()


# In[13]:


# find the movies with a negative profit and store them in a new dataframe - neg_profit
movies[movies["profit"]<0]


# # 2.3: The General Audience and the Critics

# In[14]:


movies.columns


# In[15]:


#1.	Firstly you will notice that the MetaCritic score is on a scale  so I need to change the scale of MetaCritic
movies["MetaCritic"]=movies["MetaCritic"]/10
movies


# In[16]:


#Creating a new column as "Avg_rating" by adding "IMDb_rating"&"MetaCritic" column & devided by 2
movies["Avg_rating"]=(movies["IMDb_rating"]+movies["MetaCritic"])/2


# In[17]:


#sort in decending order the "Avg_rating" column
movies1=movies[["Title","IMDb_rating","MetaCritic","Avg_rating"]]
movies1.loc[abs(movies1["IMDb_rating"]-movies1["MetaCritic"]<0.5)]


# In[18]:


#find the movies with 'MetaCritic'="IMDb_rating"<0.5 and "Avg_rating" of <8
UniversalAcclaim=movies1.loc[movies1["Avg_rating"]>=8]
movies1=movies1.sort_values(by="Avg_rating",ascending=False)
UniversalAcclaim


# # 2.4: Find the Most Popular Trios - I

# In[19]:


group=movies.pivot_table(values=["actor_1_facebook_likes","actor_2_facebook_likes","actor_3_facebook_likes"],
                  aggfunc="sum",index=["actor_1_name","actor_2_name","actor_3_name"])
group


# In[20]:


group["Total likes"]=group["actor_1_facebook_likes"]+group["actor_2_facebook_likes"]+group["actor_3_facebook_likes"]
group


# In[21]:


group.sort_values(by="Total likes",ascending=False,inplace=True)
group


# In[22]:


group.reset_index(inplace=True)
group


# In[23]:


group.iloc[0:5,:]


# # 2.5: Find the Most Popular Trios - II

# In[24]:


sorted([1,5,2])


# In[25]:


# Your answer here (optional)
j=0
for i in group["Total likes"]:
    temp=sorted([group.loc[j,"actor_1_facebook_likes"],group.loc[j,"actor_2_facebook_likes"],group.loc[j,"actor_3_facebook_likes"]])
    if temp[0]>= temp[1]/2 and temp[0]>=temp[2]/2 and temp[1]>=temp[2]/2:
        print(sorted([group.loc[j,"actor_1_name"],group.loc[j,"actor_2_name"],group.loc[j,"actor_3_name"]]))

    j=j+1


# # 2.6: Runtime Analysis

# In[26]:


#Plot a histogram or distplot of seaborn to find the Runtime range most of the movies fall into
plt.hist(movies["Runtime"])
plt.show()


# # 2.7: R-Rated Movies

# In[27]:


#Although R rated movies 
movies.loc[movies["content_rating"]=="R"].sort_values(by="CVotesU18",ascending=False)[["Title","CVotesU18"]].head(10)


# # 3 : Demographic analysis

# In[28]:


#1.First create a new dataframe df_by_genre that contains genre_1, genre_2, and genre_3 and all the columns related to CVotes/Votes from the movies data frame. There are 47 columns to be extracted in total.
df_by_genre=movies.loc[:,"CVotes10":"VotesnUS"]
df_by_genre[["genre_1","genre_2","genre_3"]]=movies[["genre_1","genre_2","genre_3"]]


# In[29]:


df_by_genre


# In[30]:


#Add a column called cnt to the dataframe df_by_genre and initialize it to 1 (one)
df_by_genre["cnt"]=1
df_by_genre


# In[31]:


df_by_genre[["genre_1","genre_2","genre_3"]]


# In[32]:


#3.	First group the dataframe df_by_genre by genre_1 
import numpy as np
df_by_g1=df_by_genre.groupby("genre_1").aggregate(np.sum)
df_by_g2=df_by_genre.groupby("genre_2").aggregate(np.sum)
df_by_g3=df_by_genre.groupby("genre_3").aggregate(np.sum)


# In[33]:


df_by_g1


# In[34]:


df_by_g2


# In[35]:


df_by_g3


# In[36]:


#add the three dataframes and store it in a new dataframe name "df_add"
df_add=df_by_g1.add(df_by_g2,fill_value=0)
df_add=df_add.add(df_by_g3,fill_value=0)
df_add


# # The column cnt on aggregation has basically kept the track of the number of occurences of each genre.Subset the genres that have atleast 10 movies into a new dataframe genre_top10 based on the cnt column value.

# In[37]:


genre_top_10=df_add.loc[df_add["cnt"]>10]
genre_top_10


# # Take the mean of all the numeric columns by dividing them with the column value cnt and store it back to the same dataframe. 

# In[38]:


genre_top_10.iloc[:,0:-1]=genre_top_10.iloc[:,0:-1].divide(genre_top_10["cnt"],axis=0)


# In[39]:


genre_top_10


# In[40]:


#Round off all the Votes related columns upto two digits after the decimal point.
genre_top_10.loc[:,"VotesM":"VotesnUS"]=round(genre_top_10.loc[:,"VotesM":"VotesnUS"],2)


# In[41]:


genre_top_10


# In[42]:


#all the CVotes related columns to integers. 
genre_top_10[genre_top_10.loc[:,"CVotes10":"CVotesnUS"].columns]=genre_top_10[genre_top_10.loc[:,"CVotes10":"CVotesnUS"].columns].astype(int)


# In[43]:


genre_top_10


# In[44]:


#Make a bar chart plotting different genres vs cnt using seaborn.
sns.barplot(x=genre_top_10["cnt"],y=genre_top_10.index)
plt.show()


# In[45]:


#1.	Make the first heatmap to see how the average number of votes of males is varying across the genres. Use seaborn heatmap for this analysis. 
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
ax=sns.heatmap(genre_top_10[["CVotesU18M","CVotes1829M","CVotes3044M","CVotes45AM"]])
plt.subplot(1,2,2)

ax=sns.heatmap(genre_top_10[["CVotesU18F","CVotes1829F","CVotes3044F","CVotes45AF"]])
plt.show()


# In[46]:


#1.	Make the first heatmap to see how the average number of votes of males is varying across the genres. Use seaborn heatmap for this analysis. 
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
ax=sns.heatmap(genre_top_10[["CVotesU18M","CVotes1829M","CVotes3044M","CVotes45AM"]],annot=True,cmap="coolwarm")
plt.subplot(1,2,2)

ax=sns.heatmap(genre_top_10[["CVotesU18F","CVotes1829F","CVotes3044F","CVotes45AF"]],annot=True,cmap="coolwarm")
plt.show()


# Inferences: A few inferences that can be seen from the heatmap above is that males have voted more than females, and Sci-Fi appears to be most popular among the 18-29 age group irrespective of their gender. What more can you infer from the two heatmaps that you have plotted? Write your three inferences/observations below:
# 
# Inference 1: Genre romance has got the least number of votes among any age group of males, but there is no such pattern among the females.
# Inference 2:Action seems to be the more popular genre among the under 18 males, and Animation appears to be the most popular genre among under 18 females.
# Inference 3: 18-29 age group seems to be most actively voting for any genre irrespective of gender

# In[47]:


#2.	Make the second heatmap to see how the average number of votes of females is varying across the genres. Use seaborn heatmap for this analysis. The X-axis should contain the four age-groups for females, i.e., CVotesU18F,CVotes1829F, CVotes3044F, and CVotes45AF. The Y-axis will have the genres and the annotation in the heatmap tell the average number of votes for that age-female group. 
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
ax=sns.heatmap(genre_top_10[["VotesU18M","Votes1829M","Votes3044M","Votes45AM"]],annot=True,cmap="coolwarm")
plt.subplot(1,2,2)

ax=sns.heatmap(genre_top_10[["VotesU18F","Votes1829F","Votes3044F","Votes45AF"]],annot=True,cmap="coolwarm")
plt.show()


# **`Inferences:`** Sci-Fi appears to be the highest rated genre in the age group of U18 for both males and females. Also, females in this age group have rated it a bit higher than the males in the same age group. What more can you infer from the two heatmaps that you have plotted? Write your three inferences/observations below:
# - Inference 1:The rating among males, seems to be decreasing with increasing age group. there is a similar pattern among females but with a few exceptions.
# - Inference 2:Crime Gener has the secong higher rating among U18 age group of both males and females, but among U18 Females, It has got the least rating.
# - Inference 3:Romance Gener has got the least rating among both 45 above Males and Females

# # Subtask 3.4: US vs non-US Cross Analysis

# In[48]:


movies["Country"].value_counts()


# In[49]:


#Creating IFUS Column
movies["IFUS"]=movies["Country"].copy()




# In[50]:


movies.loc[movies["IFUS"]!="USA","IFUS"]="non-USA"


# In[51]:


movies["IFUS"].value_counts()


# In[52]:


# 1_Box plot:- CVotesUS(y) vs IFUS(x)
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.boxplot(x="IFUS",y="CVotesUS",data=movies)
plt.subplot(1,2,2)
sns.boxplot(x="IFUS",y="CVotesnUS",data=movies)
plt.show()


# Inferences: Write your two inferences/observations below:
# 
# Inference 1:In general US movies have got the high number of vots from both US & Non-US voters when we compare the medians of Box Plot.
# 
# Inference 2:Non-US movies have a more uniform of distribution of the number of vots as compare to the US movies which is evident from the values of the quartiles.

# In[53]:


# 2_Box plot:- VotesUS(y) vs IFUS(x)
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.boxplot(x="IFUS",y="VotesUS",data=movies)
plt.subplot(1,2,2)
sns.boxplot(x="IFUS",y="VotesnUS",data=movies)
plt.show()


# # Inferences: Write your two inferences/observations below:
# 
# Inference 1:Non US voters have rated both the US & Non-US movies lower compared to the US voters, which is evident from the medians of the quartiles.
# Inference 2:US movies have received  higher ratings fron US voters
# Inference 3: Some US movies have got exceptionally high ratings from both the USA and Non-US voters. There are no such extreme ratings for any of the Non-US movies.

# -  ###  Subtask 3.5:  Top 1000 Voters Vs Genres
# 
# You might have also observed the column `CVotes1000`. This column represents the top 1000 voters on IMDb and gives the count for the number of these voters who have voted for a particular movie. Let's see how these top 1000 voters have voted across the genres. 
# 
# 1. Sort the dataframe genre_top10 based on the value of `CVotes1000`in a descending order.
# 
# 2. Make a seaborn barplot for `genre` vs `CVotes1000`.
# 
# 3. Write your inferences. You can also try to relate it with the heatmaps you did in the previous subtasks.
# 
# 
# 

# In[57]:


# Sorting by CVotes1000
genre_top_10=genre_top_10.sort_values("CVotes1000",ascending=False)
genre_top_10


# In[63]:


genre_top_10["CVotes1000"]


# In[67]:


#Bar Plt
plt.figure(figsize=(12,5))
sns.barplot(genre_top_10.index,genre_top_10["CVotes1000"])
plt.show()

# Inferences: Write your two inferences/observations below:

Inference 1:The voting pattern here almost ressembles the pattern in age group vs genre heat maps 
Inference 2: Although drama genre has the highest number of movies, the average number of top users who have rated it less as compare to other genre, Which have lesser number of movies
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




