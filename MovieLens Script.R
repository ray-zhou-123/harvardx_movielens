# Load Required Packages
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringi)) install.packages("stringi", repos = "http://cran.us.r-project.org")
library(ggplot2)
library(lubridate)
library(stringi)

# Data Cleaning - run the provided code to load and clean the dataset
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Examine the data
head(edx)
glimpse(edx) # use dplyr glimpse() to see columns
n_distinct(edx$movieId)
n_distinct(edx$genres)
n_distinct(edx$userId)

# Graph the number of ratings by userId
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, fill  = "red", color = "black") +
  scale_x_log10()+ 
  ggtitle("Number of Ratings by userId")

# Graph the number of ratings by movieId
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, fill  = "red", color = "black") +
  scale_x_log10() + 
  ggtitle("Number of Ratings by movieId")

# Generate time from timestamp useing the stringi and regex
edx <- mutate(edx, age = 2021-as.numeric(stri_extract(str_sub(edx$title,-5,-1), regex = "(\\d{4})", comments = TRUE)))
validation <- mutate(validation, age = 2021-as.numeric(stri_extract(str_sub(validation$title, -5, -1), regex =  "(\\d{4})", comments = TRUE)))

# Generate scatterplot for age versus rating
edx %>%
  group_by(age) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(age, rating)) +
  labs(title="Movie Age vs Rating", x="Age", y="Rating") +
  geom_point()

# Graph number of ratings in each rating category
qplot(as.vector(edx$rating)) +
  ggtitle("Number of Ratings of Each Rating")

# Graph the distribution of rankings by genre
edx %>% 
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(genre_rating = mean(rating)) %>%
  arrange(desc(genre_rating)) %>%
  ggplot(aes(reorder(genres, genre_rating), genre_rating)) +
  geom_bar(stat = "identity") +
  ggtitle("Highest Average Rating per Genre")

# Define method to generate RMSE to measure modeling effectiveness
RMSE <- function(model, observed){
  sqrt(mean((model-observed)^2))
}

# Model 1 - Mean + Movie Effect

d <- mean(edx$rating) # Calculate mean
d

# Generate histogram of movie effect
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(movie_effect = mean(rating-d))
b_i %>% qplot(movie_effect, geom = "histogram", data = ., fill = I("red"), col = I("black"), main = "Distribution of Movie Effect", xlab = "Movie Effect", ylab = "Count")

# Apply effect to the validation set
predicted_ratings_model_1 <- d + validation %>%
  left_join(b_i, by = "movieId") %>%
  pull(movie_effect)

# Calculate RMSE to determine the accuracy of the model
rmse_model_1 <- RMSE(predicted_ratings_model_1, validation$rating)
rmse_model_1

# Model 2 - Mean + Movie Effect + User Effect

# Generate histogram of user effect
b_u <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(user_effect = mean(rating-d-movie_effect))
b_u %>% qplot(user_effect, geom ="histogram", data = ., fill = I("red"), col=I("black"), main = "Distribution of User Effect", xlab = "User Effect", ylab = "Count")

# Apply effect to the validation set
predicted_ratings_model_2 <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(prediction = d + movie_effect + user_effect)

# Calculate RMSE to determine the accuracy of the model
rmse_model_2 <- RMSE(predicted_ratings_model_2$prediction, validation$rating)
rmse_model_2

# Model 3 - Mean + Movie + User + Age effects

# Generate histogram of age effect
b_a <- edx %>% 
  left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>%
  group_by(age) %>%
  summarize(age_effect = mean(rating-d-movie_effect-user_effect))
b_a %>% qplot(age_effect, geom ="histogram", data = ., fill = I("red"), col=I("black"), main = "Distribution of Age Effect", xlab = "Age Effect", ylab = "Count")

# Apply effect to the validation set
predicted_ratings_model_3 <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_a, by = "age") %>%
  mutate(prediction = d + movie_effect + user_effect + age_effect)

# Calculate RMSE to determine the accuracy of the model
rmse_model_3 <- RMSE(predicted_ratings_model_3$prediction, validation$rating)
rmse_model_3

# Model 4 - Regression tree

# Apply non linear method of classification and regression trees (CART)
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
library(rpart)  # Install rpart

# Build a rpart decision tree, control continues to split if the increase in r^2 is greater than the complexity parameter cp
model_4 <- rpart(rating ~ userId + movieId + age, control=rpart.control(cp=.001), data = edx)
bestcp <- model_4$cptable[which.min(model_4$cptable[,"xerror"]),"CP"] # Identify the best complexity parameter
model_4 <- prune(model_4, cp = bestcp) # Produce pruned tree based on the best complexity parameter

# Plot the splits of the tree using the rpart.plot package.
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
library(rpart.plot) # Install rpart.plot
prp(model_4, roundint=F, digits=3) # Plot the rpart model using the function in rpart.plot

# Apply model to predict ratings for the validation data
dt_pred <- predict(model_4, newdata=validation)
rmse_model_4 <- RMSE(dt_pred, validation$rating) # Generate RMSE
model_4_accuracy

# Model 5 - Matrix Factorization Using Recosystem

if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
library(recosystem) # Install recosystem

r <- Reco()
train_data <- data_memory(edx$userId, edx$movieId, edx$rating)
test_data <- data_memory(validation$userId, validation$movieId, validation$rating)
mf_pred <- r$predict(test_data, out_memory(), opts = list(dim = 20, verbose = FALSE))
rmse_model_5 <- RMSE(mf_pred, validation$rating)
rmse_model_5

# Generate table of results
tab <- matrix(c(rmse_model_1, rmse_model_2, rmse_model_3, rmse_model_4, rmse_model_5), ncol = 1, byrow = TRUE) # Accumulate all results in one col matrix
colnames(tab) <- "Root Mean Squared Error"
rownames(tab) <- c("Movie Effect", "User + Movie Effect", "User + Movie + Age Effect", "Decision Tree", "Matrix Factorization")
tab <- as.table(tab)
tab