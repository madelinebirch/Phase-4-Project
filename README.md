# CineSYNC: Navigating the Stream to Success
**An Exploratory Approach to Personalized Recommendation Systems**

<img src="Images/cineSYNC_logo.png" alt="CineSYNC Logo" width="300" height="200">

# Business Understanding
### Our Primary Stakeholder
Our protagonist in this story is cineSYNC, a veteran streaming service that's resurrecting itself out of obscurity. Established in 2006 and operating as "the people's streaming service" ever since, the platform's many users celebrate the outdated, low-tech simplicity of its straightforward movie library and basic UI akin to that of vintage piracy sites like Levidia.ch or Project Free TV. Bolstered by recent investments, cineSYNC has set its sights on a major transformation, aiming to provide a more personalized user experience competitive with industry heavyweights like Netflix and HBO Max, while maintaining their integrity as a universally approachable brand and affordable service. Deviating from their longstanding ads-only revenue model, cineSYNC is prepared to offer low-cost subscriptions in exchange for user profiles, rating options, and customized title recommendations, recognizing that the key to success lies in aligning with user tastes and meeting user preferences. 

### Our Mission
Our mission is to introduce cineSYNC to the basics and intricacies of user-based recommendation systems. We'll delve into various collaborative filtering techniques such as Singular Value Decomposition (SVD) and Alternating Least Squares (ALS), demystify the underlying concepts, and evaluate model performance. Collaborative filtering, a key focus of our exploration, involves recommending items based on the preferences and behavior of similar users. This approach, in contrast to content-based or hybrid filtering systems, harnesses the collective wisdom of the user community, allowing cineSYNC to tap into the diverse tastes of its audience.

<img src="Images/cf_diagram.png" alt="Collaborative Filtering" width="800" height="600">
[Image source: https://medium.com/@ashmi_banerjee/understanding-collaborative-filtering-f1f496c673fd]


#### Why Collaborative Filtering?
Collaborative filtering holds a distinct advantage in capturing complex and evolving user preferences. Unlike content-based approaches, which rely on item characteristics, collaborative filtering adapts dynamically to user behavior, making it well-suited for platforms like cineSYNC with a vast and diverse content library. By leveraging the wisdom of the crowd, collaborative filtering facilitates personalized recommendations, enhancing user satisfaction and engagement.


# Data Understanding
<img src="Images/grouplens_logo.png" alt="grouplens Logo" width="300" height="200">

In this project, we will be working with the MovieLens dataset provided by the GroupLens research lab at the University of Minnesota. The dataset is a well-known resource in the field of recommendation systems and contains information about movies, user ratings, and user-generated tags.

### Dataset Components
The dataset is distributed in the ml-latest-small.zip archive and contains the following four CSV files:

**`movies.csv:`** This file contains information about movies, including their unique identifiers (movieId), titles, and genres. It is an essential part of our recommendation system as it provides details about the movies that users have rated.

**`ratings.csv:`** This file contains user ratings for various movies. It includes the user's unique identifier (userId), the movie's unique identifier (movieId), the rating they assigned to the movie, and a timestamp indicating when the rating was recorded. This dataset will be the primary source for building our recommendation system based on collaborative filtering.

**`tags.csv:`** This file contains user-generated tags for movies. Each entry includes the userId, movieId, the tag text, and a timestamp. While not the focus of our exploration in collaborative filtering, it is worth highlighting here for the purposes of cineSYNC's future content-based or hybrid modeling endeavors.

**`links.csv:`** This file contains links to external movie databases such as IMDb and TMDb, using identifiers for movieId. This external data could be useful to cineSYNC, but do not fit within the scope of our project.


# Data Preparation

*See notebook for a full EDA and plotted distributions of 'ratings' and other engineered features.* 

**For modeling, will be working with a cleaned and filtered version of `ratings_df`, merged with a one-hot encoded `movies_df` as `movies_and_ratings_df` for ease of use in future content-based and/or hybrid modeling.** 

# Modeling

### Target
Because we are using collaborative filtering, our prediction target will be the `'ratings'` feature from `ratings_df`.

### Evaluation Metrics
#### RMSE (Root Mean Squared Error):
RMSE measures the average magnitude of the differences between predicted and actual values, emphasizing larger errors. For cineSYNC, a lower RMSE indicates better predictive accuracy and alignment with user preferences. The closer to 0, the more accurate.

<img src="Images/RMSE_equation.png" alt="RMSE" width="300" height="200">
[Image source: https://docs.oracle.com/en/cloud/saas/planning-budgeting-cloud/pfusu/img/insights_rmse_formula.jpg]

#### MAE (Mean Absolute Error):
MAE gauges the average absolute differences between predicted and actual values, providing a more straightforward measure of model performance. Similar to RMSE, a lower MAE signifies better accuracy in predicting user preferences.

<img src="Images/MAE_equation.png" alt="MAE" width="300" height="200">
[Image source: https://medium.com/@polanitzer/the-minimum-mean-absolute-error-mae-challenge-928dc081f031]

### Model 1: Alternating Least Squares (ALS) with PySpark

#### What is ALS?
To begin our journey into cineSYNC's potential with CF systems, we'll firstly introduce the ALS model. The ALS (Alternating Least Squares) recommendation model is a supervised learning algorithm commonly employed in recommendation systems. Unlike traditional supervised learning, where the algorithm learns from labeled data with explicit input-output pairs, ALS operates in a unique manner within the collaborative filtering framework.

In supervised learning, the algorithm learns a mapping from input features to target labels based on a labeled training dataset. In contrast, ALS is employed in a semi-supervised fashion for collaborative filtering, which inherently lacks explicit labels. Instead, it utilizes the implicit feedback inherent in user-item interactions, such as user ratings, to capture underlying patterns in the data.

The ALS algorithm optimizes the latent factors by iteratively decomposing the user-item interaction matrix, minimizing the reconstruction error between observed and predicted ratings. This iterative optimization process is the hallmark of ALS, as it alternates between fixing one matrix (either user or item factors) and optimizing the other. Through this alternating process, ALS uncovers latent features that represent user preferences, facilitating the generation of accurate and personalized recommendations.

#### How is PySpark Used in the Context of ALS?
**PySpark,** the Python API for Apache Spark, offers a scalable and distributed computing framework ideal for large-scale collaborative filtering tasks. ALS in PySpark facilitates the training of models on extensive datasets, ensuring efficient handling of user-item interactions. This is particularly valuable for cineSYNC's ambition to provide a more personalized user experience at scale.

*See notebook for full code and markdown of ALS modeling.*
<img src="Images/ALS_rec_output.png" alt="ALS Top 10" width="400" height="300">


