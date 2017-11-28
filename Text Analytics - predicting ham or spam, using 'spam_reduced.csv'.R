#Text Analytics - predicting ham/spam, using 'spam_reduced.csv'
getwd()
rm(list =ls())

#read file
spam.raw <- read.csv("spam_reduced.csv", stringsAsFactors = F, fileEncoding = "UTF-16")
View(spam.raw)
str(spam.raw)
summary(spam.raw)
sum(!complete.cases(spam.raw)) #no NA's

#selecting required features
spam.raw <- spam.raw[, 2:3]
View(spam.raw)

# Convert our 'Label' into a factor.
spam.raw$Label <- as.factor(spam.raw$Label)
prop.table(table(spam.raw$Label))

#Adding a feature
spam.raw$TextLength <- nchar(spam.raw$Text)
summary(spam.raw$TextLength)
str(spam.raw)

# Visualize distribution with ggplot2, adding segmentation for ham/spam.
library(ggplot2)
ggplot(spam.raw, aes(x = TextLength, fill = Label)) +
  theme_dark() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")

#stratified split into train/test
library(caret)
set.seed(32984)
indexes <- createDataPartition(spam.raw$Label, times = 1,p = 0.7, list = F)

train <- spam.raw[indexes,]
test <- spam.raw[-indexes,]

prop.table(table(train$Label))
prop.table(table(test$Label))

#invoking quanteda to clean the dataset
library(quanteda)
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = T, remove_punct = T,
                       remove_symbols = T, remove_hyphens = T)
train.tokens <- tokens_tolower(train.tokens)
train.tokens <- tokens_select(train.tokens, stopwords(), selection = "remove")
train.tokens <- tokens_wordstem(train.tokens, language = "english")

# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = F)
View(train.tokens.dfm[1:20, 1:100])
dim(train.tokens.dfm)

# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100])
dim(train.tokens.matrix)

# cbind 'Label' to the dataframe
length(train$Label)
dim(train.tokens.dfm) #no of rows = train$Label, hence cbind can be done
train.tokens.df <- cbind(Label = train$Label, as.data.frame(train.tokens.dfm))
View(train.tokens.df[1:10,1:10])
dim(train.tokens.df) #an extra column hence '1992'

# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))

#Cross-Validation: 10Folds repeated 3times = 30 random samples
library(caret)
set.seed(48743)
cv.folds <- createMultiFolds(train$Label, k = 10, times = 3)
cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)
#train: Using 'rpart' - a single decision tree, as it's faster
library(doSNOW)
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
rpart.cv.1 <- train(Label ~ ., data = train.tokens.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
rpart.cv.1 #95.17% accuracy

#*****************************************************************************

#Building TF-IDF:

#TF-IDF score of a word, ranks it's importance.
#tfidf score of a word = tf*idf
#where, tf : No. of times that word appear in document/Total words in document
#and, idf : log(No. of documents/No. of documents that contain that word)

#    1 - The TF calculation accounts for the fact that longer 
#        documents will have higher individual term counts. Applying
#        TF normalizes all documents in the corpus to be length 
#        independent.
#    2 - The IDF calculation accounts for the frequency of term
#        appearance in all documents in the corpus. The intuition 
#        being that a term that appears in every document has no
#        predictive power.
#    3 - The multiplication of TF by IDF for each cell in the matrix
#        allows for weighting of #1 and #2 for each cell in the matrix.

#TF:
tf <- function(row){row/sum(row)} #note: this transposes the matrix

#IDF:
idf <- function(col){
corpus.size <- length(col)
doc.count <- length(which(col > 0))
log10(corpus.size / doc.count)
}

#TF*IDF:
tf.idf <- function(tf,idf){tf * idf}


#Setp1. Apply TF:
train.tokens.df <- apply(train.tokens.matrix, 1, tf) #returns matrix
dim(train.tokens.df) #note the transpose
View(train.tokens.df[1:20, 1:20])

#Step2. Apply IDF:
train.tokens.idf <- apply(train.tokens.matrix, 2, idf) #returns numeric
head(train.tokens.idf)
View(head(train.tokens.idf))

#Step3. Apply TF*IDF:
train.tokens.tfidf <- apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
class(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:20, 1:20])

#Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:20, 1:20])

#Check for incopmlete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]

#Fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
sum(which(!complete.cases(train.tokens.tfidf)))

#cbind 'Label' variable to finalize our dataframe:
train.tokens.tfidf.df <- cbind(Label = train$Label, as.data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

#train: Using 'rpart' again
library(doSNOW)
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
rpart.cv.2 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
rpart.cv.2 #95.30% accuracy
#*****************************************************************************

#N-grams allow us to augment our document-term frequency matrices with
#word ordering. This often leads to increased performance (e.g., accuracy)
#for machine learning models trained with more than just unigrams (i.e.,
#single terms). Let's add bigrams to our training data and the TF-IDF 
#transform the expanded featre matrix to see if accuracy improves.

#Add bigrams:
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)
train.tokens[[357]] #see the difference of using bi-grams

#Transform to dfm and then a matrix.
train.tokens.dfm <- dfm(train.tokens, tolower = F)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
dim(train.tokens.dfm) #note, how the no.of terms have exploded to 6476 from 1992

#Apply TF:
train.tokens.df <- apply(train.tokens.matrix, 1, tf)
#Apply IDF:
train.tokens.idf <- apply(train.tokens.matrix, 2, idf)
#Apply TF*IDF:
train.tokens.tfidf <- apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)

#Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)

#Fix incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))

#cbind 'Label' variable to finalize our dataframe:
train.tokens.tfidf.df <- cbind(Label = train$Label,
                               as.data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))
#Clean up unused objects in memory.
gc()

#train: Use 'rpart' third time, as it is faster than 'rf'
library(doSNOW)
set.seed(94156)
cl <- makeCluster(1, type = 'SOCK')
registerDoSNOW(cl)
rpart.cv.3 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
rpart.cv.3 #95.30% accuracy

#******************************************************************************
#SVD:Singular value decomposition

#Use 'irlba' for SVD. 'irlba' allows us to specify
#the number of the most important singular vectors we wish to
#calculate and retain for features.
library(irlba)
#SVD reduces dimensionality down to 300 columns for LSA.
train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)
View(train.irlba$v)

# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
train.irlba$v[1, 1:10]
document.hat[1:10] #notice the approximation

# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
train.svd <- data.frame(Label = train$Label, train.irlba$v)

#train: Use 'rpart', fourth and final time
library(doSNOW)
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
rpart.cv.4 <- train(Label ~ ., data = train.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
rpart.cv.4 #92.6% accuracy (reduction in dimendions, caused dip in accuracy)

#******************************************************************************
#train: Using 'rf' finally
#(10folds * 3times * 7mtry(tuneLength) * 500(default 'rf' trees)) + 500 (final group of trees) = 105,500 trees
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
rf.cv.1 <- train(Label ~ ., data = train.svd, method = "rf",
                trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)

rf.cv.1 #95.44% accuracy
confusionMatrix(train.svd$Label, rf.cv.1$finalModel$predicted)

#Adding 'TextLength':
train.svd$TextLength <- train$TextLength

#train: Using 'rf' with importance=T
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
rf.cv.2 <- train(Label ~ ., data = train.svd, method = "rf",
                 trControl = cv.cntrl, tuneLength = 7,
                 importance = T)
stopCluster(cl)

rf.cv.2
confusionMatrix(train.svd$Label, rf.cv.2$finalModel$predicted)

# How important was the new feature?
library(randomForest)
varImpPlot(rf.cv.1$finalModel)
varImpPlot(rf.cv.2$finalModel) #note the importance of 'TextLength'

#***************************************************************************
#Cosine Similarity:

#Cosine Similarity to engineer a feature,for calculating, on average,
#how alike each Text is, to all of the spam messages.

#The hypothesis here is that, our use of bigrams, tf-idf, and LSA,
#have produced a representation where ham Text should have low cosine
#similarities with spam Text, and vice versa.

#Invoking lsa's cosine function:
library(lsa)
train.similarities <- cosine(t(as.matrix(train.svd[, -c(1, ncol(train.svd))])))

#Next up - take each Text and find what the mean cosine 
#similarity is for each Text with each of the spam Text.
#Per our hypothesis, ham Text should have relatively low
#cosine similarities with spam Text and vice versa!
spam.indexes <- which(train$Label == "spam")

train.svd$SpamSimilarity <- rep(0.0, nrow(train.svd))
for(i in 1:nrow(train.svd)) {
  train.svd$SpamSimilarity[i] <- mean(train.similarities[i, spam.indexes])  
}

#Visualize:
library(ggplot2)
ggplot(train.svd, aes(x = SpamSimilarity, fill = Label)) +
  theme_dark() +
  geom_histogram(binwidth = 0.05) +
  labs(y = "Message Count",
       x = "Mean Spam Message Cosine Similarity",
       title = "Distribution of Ham vs. Spam Using Spam Cosine Similarity")

#train: Using 'rf' for Cosine Similarity data
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
set.seed(932847)
rf.cv.3 <- train(Label ~ ., data = train.svd, method = "rf",
                 trControl = cv.cntrl, tuneLength = 7,
                 importance = T)
stopCluster(cl)

rf.cv.3
confusionMatrix(train.svd$Label, rf.cv.3$finalModel$predicted)

# How important was this feature?
library(randomForest)
varImpPlot(rf.cv.3$finalModel)


#****************************************************************************
#   Now, since we've trained our final model, we'd run the same for 'test'
#   Since, every new dataset has to follow the same 'Vector Space', hence:
#****************************************************************************
#      Step.1 - Tokenize
#      Step.2 - Apply TF, and TF*IDF
#      Step.3 - Fix incomplete cases
#      Step.4 - SVD Projections.
#      Step.5 - Prepare final svd df, by adding 'Label' and 'TextLength'.
#      Step.6 - Forming cosine similarity
#      Step.7 - Predict results prior training (tentative results)
#      Step.8 - final train the model
#      Step.9 - final results after training
#****************************************************************************

#Step.1 - Tokenize:
test.tokens <- tokens(test$Text, what = "word", 
                      remove_numbers = T, remove_punct = T,
                      remove_symbols = T, remove_hyphens = T)
test.tokens <- tokens_tolower(test.tokens)
test.tokens <- tokens_select(test.tokens, stopwords(), 
                             selection = "remove")
test.tokens <- tokens_wordstem(test.tokens, language = "english")
test.tokens <- tokens_ngrams(test.tokens, n = 1:2)
test.tokens.dfm <- dfm(test.tokens, tolower = F)
#Ensure the test dfm has the same n-grams as the training dfm.
test.tokens.dfm <- dfm_select(test.tokens.dfm, pattern = train.tokens.dfm,
                              selection = "keep")
test.tokens.matrix <- as.matrix(test.tokens.dfm)

dim(train.tokens.dfm)
dim(test.tokens.dfm)

#Step2. Apply TF,and TF*IDF
test.tokens.df <- apply(test.tokens.matrix, 1, tf)
str(test.tokens.df)

test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(test.tokens.tfidf)
View(test.tokens.tfidf[1:25, 1:25])

test.tokens.tfidf <- t(test.tokens.tfidf) #transpose

#Step3. Fix incomplete cases
summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)] <- 0.0
summary(test.tokens.tfidf[1,])

#Step4. SVD projection:
test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf))

#Step5. Prepare final svd df, by adding 'Label' and 'TextLength'.
test.svd <- data.frame(Label = test$Label, test.svd.raw, 
                       TextLength = test$TextLength)

#Step6. Forming cosine similarity
test.similarities <- rbind(test.svd.raw, train.irlba$v[spam.indexes,])
test.similarities <- cosine(t(test.similarities))

test.svd$SpamSimilarity <- rep(0.0, nrow(test.svd))
spam.cols <- (nrow(test.svd) + 1):ncol(test.similarities)
for(i in 1:nrow(test.svd)) {
  test.svd$SpamSimilarity[i] <- mean(test.similarities[i, spam.cols])  
}

# Some SMS text messages become empty as a result of stopword and special 
# character removal. This results in spam similarity measures of 0.
test.svd$SpamSimilarity[!is.finite(test.svd$SpamSimilarity)] <- 0


#Step7. Predict results prior training (tentative results):
preds <- predict(rf.cv.3, test.svd)
confusionMatrix(preds, test.svd$Label)

#Step8. final train the model
library(doSNOW)
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
set.seed(254812)
rf.cv.4 <- train(Label ~ ., data = train.svd, method = "rf",
                 trControl = cv.cntrl, tuneLength = 7,
                 importance = T)
stopCluster(cl)

#Step9. final results after training:
preds <- predict(rf.cv.4, test.svd)
confusionMatrix(preds, test.svd$Label)
