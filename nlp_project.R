# Install required packages
install.packages("ggplot2")
install.packages("RColorBrewer")
install.packages("wordcloud")
install.packages("ggraph")
install.packages("tidyverse")
install.packages("tidytext")
install.packages("tm")
install.packages("dplyr")
install.packages("reshape2")
install.packages("syuzhet")
install.packages("e1071")
install.packages("naivebayes")
install.packages("caret")
install.packages("randomForest")
install.packages("qdap")

# Load required packages
library(ggplot2)
library(RColorBrewer)
library(wordcloud)
library(ggraph)
library(tidyverse)
library(tidytext)
library(tm)
library(dplyr)
library(reshape2)
library(syuzhet)
library(e1071)
library(naivebayes)
library(caret)
library(randomForest)
library(qdap)

# Reading the data
covid_train <- read_csv("Corona_NLP_train.csv")  
covid_test <- read_csv("Corona_NLP_test.csv")  

covid_data <- rbind(covid_train, covid_test) # Full dataset

# Because of I cannot deal with the size of the dataset and the memory space in my computer 
# I have had to decrease the number of observations to 5000 random rows
covid_data <- na.omit(covid_data)
covid_data <- covid_data[sample(nrow(covid_data), 5000), ]


# -----------------------EDA--------------------------------
# How many Sentiments in each category of Sentiments

ggplot(data = covid_data, mapping = aes(x = Sentiment, fill = as.factor(Sentiment))) + 
  geom_bar() +
  labs(x = "Sentiment", y = "Frequency", fill = "Sentiment")


################################################################################

# Let's set the sentiments into 3 different groups:
  # Positive: "Extremely Positive" and "Positive"
  # Negative: "Extremely Negative" and "Negative"
  # Neutral: "Neutral"

covid_data$Sentiment[covid_data$Sentiment == "Extremely Positive"] <- "Positive"
covid_data$Sentiment[covid_data$Sentiment == "Extremely Negative"] <- "Negative"


# Plotting those three sentiments
ggplot(data = covid_data, mapping = aes(x = Sentiment, fill = as.factor(Sentiment))) + 
  geom_bar() + 
  labs(x = "Sentiment", y = "Frequency", fill = "Sentiment")


################################################################################


# counting rows (Number of rows by each sentiment to calculate the percentage)
num_sentiment_data <- covid_data %>% 
  group_by(Sentiment) %>% 
  tally()


################################################################################

# -----------------------Cleaning data-----------------------------

# ------------ Cleaning functions ------------
clean_tweets <- function(p.tweets) {
  p.tweets <- tolower(p.tweets)
  p.tweets <- gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", p.tweets)
  p.tweets <- gsub("@\\S+", "", p.tweets)
  p.tweets <- gsub("http\\S*", "", p.tweets)
  p.tweets <- gsub("[^a-zA-Z0-9 ]", "", p.tweets)
  p.tweets <- gsub("[[:punct:]]", "", p.tweets)
  p.tweets <- gsub("[[:digit:]]", "", p.tweets)
  p.tweets <- gsub("amp", "", p.tweets)
  p.tweets <- gsub("\\btco[a-zA-Z0-9]*\\b", "", p.tweets)
  p.tweets <- gsub("[ \t]{2,}", "", p.tweets)
  p.tweets <- gsub("^\\s+|\\s+$", "", p.tweets)
  p.tweets <- iconv(p.tweets, 'UTF-8', 'ASCII')
  return(p.tweets)
}

clean_corpus <- function (p.corpus) {
  p.corpus <- tm_map(p.corpus, content_transformer(tolower))
  p.corpus <- tm_map(p.corpus, removeNumbers)
  p.corpus <- tm_map(p.corpus, removePunctuation)
  p.corpus <- tm_map(p.corpus, stripWhitespace)
  p.corpus <- tm_map(p.corpus, removeWords, c(stopwords("en"), "get", "just", "will",
                                              "now", "thats", "lot", "let", "lets", "etc", "don",
                                              "can", "put", "people"))
  return(p.corpus)
}
# ---------------------------------------------
tweets <- clean_tweets(covid_data$OriginalTweet)

# Let's create the Corpus
corpus <- VCorpus(VectorSource(tweets))

# Cleaning the corpus
tweets_corpus <- clean_corpus(corpus)
tweets_tdm <- TermDocumentMatrix(tweets_corpus)
tweets_tdm <- removeSparseTerms(tweets_tdm, 0.995)


tweets_mtrx <- as.matrix(tweets_tdm)
words <- sort(rowSums(tweets_mtrx), decreasing = TRUE)
words_df <- data.frame(word = names(words), freq = words)


# Wordcloud
set.seed(1234)
wordcloud(words = words_df$word, freq = words_df$freq,
          max.words = 160, random.order = FALSE, rot.per = 0.35,
          colors = brewer.pal(8, "Dark2"))

# Word Frequencies
barplot(words_df[1:20,]$freq, las = 2, names.arg = words_df[1:20,]$word,
        col ="cornflowerblue", main ="Top 20 frequent words",
        ylab = "Word frequencies")


################################################################################

# Words by Negative vs Positive Sentiment - Sentiment Analysis with Bing 
covid_data$OriginalTweet <- clean_tweets(covid_data$OriginalTweet)
covid_tokens <- covid_data %>% 
  select(OriginalTweet) %>% 
  unnest_tokens(word, OriginalTweet)
cleaned_tweets <- covid_tokens %>% anti_join(stop_words)
covid_sentiments <- covid_tokens %>% 
  inner_join(get_sentiments("bing")) %>% 
  count(word, sentiment, sort = TRUE) %>% 
  ungroup()

covid_sentiment_plot <- covid_sentiments %>% 
  group_by(sentiment) %>% 
  top_n(10) %>% 
  ungroup() %>% 
  mutate(word = reorder(word, n)) %>% 
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(title = "Negative vs. Positive | Sentiments - COVID19", 
       y = NULL,
       x = NULL) + 
  coord_flip() + theme_bw()


# Sentiments Score - Sentiment Analysis with Syuzhet
covid_scores <- get_nrc_sentiment(covid_data$OriginalTweet)
scores <- data.frame(colSums(covid_scores[,]))
names(scores) <- "Score"
scores <- cbind("sentiment" = rownames(scores), scores)
rownames(scores) <- NULL

ggplot(scores, aes(sentiment, Score)) +
  geom_bar(aes(fill = sentiment), stat = "identity") + 
  theme(legend.position = "none") +
  labs(title = "COVID-19 Sentiments' Scores") + 
    xlab("Sentiments") + 
    ylab("Scores")


# WordCloud by sentiments - Sentiment Analysis with Bing
covid_tokens %>%
  inner_join(get_sentiments("bing")) %>% 
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("red", "dark green"),
                   max.words = 160)

# Associations with Coronavirus
covid_assocs <- findAssocs(tweets_tdm, term = "coronavirus", corlimit = 0.05)
assocs_df <- list_vect2df(covid_assocs)[, 2:3]

ggplot(assocs_df, aes(y = assocs_df[, 1])) + 
  geom_point(aes(x = assocs_df[, 2]), 
             data = assocs_df, size = 3) + 
  ggtitle("Word Associations with Coronavirus") + 
  xlab("Correlation") + 
  ylab("Words") +
  theme_bw()

################################################################################
# --- Model Classification ---

# Split the dataset in training data and testing data
covid_data$OriginalTweet <- clean_tweets(covid_data$OriginalTweet)
covid_data$Sentiment <- as.factor(covid_data$Sentiment)

# Execute these lines twice !!!!!
indx <- createDataPartition(covid_data$Sentiment, p = 0.7, list = FALSE)
training <- covid_data[indx, ] 
testing <- covid_data[-indx, ]


# Naive Bayes Model
set.seed(123)
model_nb <- naiveBayes(Sentiment ~ OriginalTweet, data = training) # Create the Naive Bayes' model

pred_nb <- predict(model_nb, newdata = testing) # Testing the model
mtrx_nb <- table(testing$Sentiment, pred_nb)
confusionMatrix(mtrx_nb) # Accuracy 43.55%


# Random Forest Model
set.seed(456)
model_rf <- randomForest(Sentiment ~ OriginalTweet, data = training) # Create the Random Forest's model

pred_rf <- predict(model_rf, newdata = testing) # Testing the model
mtrx_rf <- table(testing$Sentiment, pred_rf)
confusionMatrix(mtrx_rf) # Accuracy 39.03%

################################################################################