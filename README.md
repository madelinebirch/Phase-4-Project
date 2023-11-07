# CineSYNC: Navigating the Stream to Success
**An Exploratory Approach to Personalized Recommendation Systems**

<img src="Images/cineSYNC_logo.png" alt="CineSYNC Logo" width="300" height="200">

# Business Understanding
### Our Primary Stakeholder
Our protagonist in this story is CineSYNC, a veteran streaming service that's resurrecting itself out of obscurity. Established in 2006 and operating as "the people's streaming service" ever since, the platform's many users celebrate the outdated, low-tech simplicity of its straightforward movie library and basic UI akin to that of vintage piracy sites like Levidia.ch or Project Free TV. Bolstered by recent investments, CineSYNC has set its sights on a major transformation, aiming to provide a more personalized user experience competitive with industry heavyweights like Netflix and HBO Max, while maintaining their integrity as a universally approachable brand and affordable service. Deviating from their longstanding ads-only revenue model, cineSYNC is prepared to offer low-cost subscriptions in exchange for user profiles, rating options, and customized title recommendations, recognizing that the key to success lies in aligning with user tastes and meeting user preferences. 

### Our Mission
Our mission is to introduce CineSYNC to the basics and intricacies of user-based recommendation systems. We'll delve into various collaborative filtering techniques such as Singular Value Decomposition (SVD) and Alternating Least Squares (ALS), demystify the underlying concepts, and evaluate model performance. Model evaluation metrics RMSE (root mean squared error) and MAE (mean absolute error) will be our compass along this exploratory journey, laying the groundwork for CineSYNC to build on and iterate further in modeling processes of their own. With our foundational support, CineSYNC can be better positioned as a leading underdog among industry giants.

# Data Understanding
<img src="Images/grouplens_logo.png" alt="grouplens Logo" width="300" height="200">

In this project, we will be working with the MovieLens dataset provided by the GroupLens research lab at the University of Minnesota. The dataset is a well-known resource in the field of recommendation systems and contains information about movies, user ratings, and user-generated tags.

### Dataset Components
The dataset is distributed in the ml-latest-small.zip archive and contains the following four CSV files:

**`movies.csv:`** This file contains information about movies, including their unique identifiers (movieId), titles, and genres. It is an essential part of our recommendation system as it provides details about the movies that users have rated.

**`ratings.csv:`** This file contains user ratings for various movies. It includes the user's unique identifier (userId), the movie's unique identifier (movieId), the rating they assigned to the movie, and a timestamp indicating when the rating was recorded. This dataset will be the primary source for building our recommendation system based on collaborative filtering.

**`tags.csv:`** This file contains user-generated tags for movies. Each entry includes the userId, movieId, the tag text, and a timestamp. While not the focus of our exploration in collaborative filtering, it is worth highlighting here for the purposes of cineSYNC's future content-based or hybrid modeling endeavors.

**`links.csv:`** This file contains links to external movie databases such as IMDb and TMDb, using identifiers for movieId. This external data could be useful to cineSYNC, but do not fit within the scope of our project.

