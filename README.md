# Not Funny!: A Joke Recommendation System 

### Tyler Watson & Kevin Magaña, Dec. 17th, 2017

## The Challenge: 
Build a recommender system based off of data from the Jester Dataset, which includes user ratings of over 100 jokes. The Berkley 
dataset includes over a million user ratings. 

## Exploratory Data Analysis 

<p align="center"> 
<img src="images/ratings.png">
</p>

## The Goal: 
The goal is to build a recommendation system to suggest jokes to users. We will score the model based off of how well we
predicted the top-rated jokes for the user ratings in our test set. 

## How: 
We implemented item similarity recommenders (using cosine, pearson, and jaccard similarity types) and factorization
recommenders using GraphLab. 

<p align="center"> 
<img src="images/item_similarity_score_vs_similarity_type.png">
</p>


<p align="center"> 
<img src="images/topk_score_80_96.png">
</p>


<p align="center"> 
<img src="">
</p>












