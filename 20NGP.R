require(tm)
require(SnowballC)
library(NbClust)
library(wordcloud)
library(lsa)
library(topicmodels)
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra)

fdir <-c(
         "/users/sharul/desktop/assignment 1/20_newsgroups/comp.sys.ibm.pc.hardware",
         "/users/sharul/desktop/assignment 1/20_newsgroups/rec.sport.hockey",
         "/users/sharul/desktop/assignment 1/20_newsgroups/soc.religion.christian")
train.data <- Corpus(DirSource(fdir, recursive=TRUE), readerControl =list(reader=readPlain))
train.data
category <- vector("character", length(train.data))

category[1:1000] <- "comp.sys.ibm.pc.hardware"
category[1001:2000] <- "rec.sport.hockey"
category[2001:2997] <- "soc.religion.christian"
#processing the data
train.data.p<-tm_map(train.data,removeWords,"Subject")
train.data.p<-tm_map(train.data.p,removeWords,"Organization")
train.data.p<-tm_map(train.data.p,removeWords,"writes")
train.data.p<-tm_map(train.data.p,removeWords,"From")
train.data.p<-tm_map(train.data.p,removeWords,"lines")
train.data.p<-tm_map(train.data.p,removeWords,"NNTP-Posting-Host")
train.data.p<-tm_map(train.data.p,removeWords,"article")
train.data.p<-tm_map (train.data.p, content_transformer(tolower))
train.data.p <- tm_map(train.data.p,removeWords,c("from","subject","summary","keywords","nntp-posting-host","nntppostinghost",
                                "organization","lines","article","writes","line","path","newsgroup","sender","messageid","date"))
train.data.p<-tm_map (train.data.p, removePunctuation)
train.data.p<-tm_map (train.data.p, stripWhitespace)
train.data.p<-tm_map (train.data.p, removeNumbers)
train.data.p<-tm_map (train.data.p, removeWords,stopwords('english'))
train.data.p<-tm_map (train.data.p, stemDocument)

tdm <- TermDocumentMatrix (train.data.p,control = list(wordLengths=c(4,20)))
inspect(tdm[1:20, 1:100])
dtm<-DocumentTermMatrix(train.data.p,control = list(wordLengths=c(4,20)))
inspect(dtm[1:20, 1:100])

mat=as.matrix(tdm)
v<-sort(rowSums(mat),decreasing = TRUE)
d<-data.frame(word=names(v),freq=v)
head(v,10)
freq<-sort(colSums(as.matrix(dtm)),decreasing = TRUE)
head(freq,15)
wordcloud(names(freq),freq,max.words=100,scale = c(3.0,0.5),color=brewer.pal(8,"Dark2"))

dtms <- removeSparseTerms(dtm, 0.95)
tfidf <- weightTfIdf(dtms)
tfidf
tdms <- removeSparseTerms(tdm, 0.95)
tdm_tfidf<-weightTfIdf(tdms)
dtm_tfidf<-weightTfIdf(dtms)
dtm_tfidf
inspect(dtm_tfidf[1:10, 100:200])
inspect(tdm_tfidf[1:10, 100:200])
m <- as.matrix(tfidf)
rownames(m) <- 1:nrow(m)

#Euclidean distance
#norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
#m_norm <- norm_eucl(m)

#k-means

k3 <- kmeans(tfidf, 3)
plot(prcomp(tfidf)$x,col=k3$cluster)
sse3 <- ((k3$tot.withinss/k3$totss)*100)
sse3
k3$totss
table(k3$cluster)
Confm <- table(category, k3$cluster)
Confm

k4 <- kmeans(tfidf, 3)
plot(prcomp(tfidf)$x,col=k4$cluster)
sse4 <- ((k4$tot.withinss/k4$totss)*100)
sse4
k4$totss
table(k4$cluster)
Confm <- table(category, k4$cluster)
Confm

k5 <- kmeans(tfidf, 4)
plot(prcomp(tfidf)$x,col=k5$cluster)
sse5 <- ((k5$tot.withinss/k5$totss)*100)
sse5
k5$totss
table(k5$cluster)
Confm <- table(category, k5$cluster)
Confm

#Elbow

set.seed(123)

# function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(tfidf, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 2:8

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
set.seed(123)

fviz_nbclust(as.matrix(tfidf), kmeans, method = "wss")

#Optimal Clusters
avg_sil <- function(k) {
  km.res <- kmeans(tfidf, centers = k, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(tfidf))
  mean(ss[, 3])
}

# Compute and plot wss for k = 2 to k = 9
k.values <- 2:9

# extract avg silhouette for 2-9 clusters
avg_sil_values <- map_dbl(k.values, avg_sil)

plot(k.values, avg_sil_values,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Average Silhouettes")
fviz_nbclust(as.matrix(tfidf), kmeans, method = "silhouette")

#nbclust

n <- NbClust(as.matrix(tfidf), distance = "euclidean", min.nc = 2, max.nc = 9, method = 'complete', index = 'silhouette')
head(n,2)

#Remove sparse items

#lda

burnin <- 4000
iter <- 1000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

k <- 5

ldaOut <-LDA(dtm,4, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

tot_prob<-as.data.frame(ldaOut@gamma)
pro<-as.matrix(tot_prob)
rownames(pro) <- 1:nrow(pro)
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
norm_pro<-norm_eucl(pro)
cl_3<-kmeans(norm_pro,k)
plot(prcomp(norm_pro)$x,col=cl_3$cluster,main='LDA for k=4')
#most representative words
ldaOut.terms <- as.matrix(terms(ldaOut,10))
ldaOut.terms

lda_data<-as.data.frame(ldaOut@gamma)
lda_mat<-as.matrix(lda_data)
rownames(lda_mat)<-1:nrow(lda_mat)
norm_lda<-norm_eucl(lda_mat)
norm_lda
# Cluster with kMeans
lda_cl <- kmeans(norm_lda, 3)
lda_Confm <- table(category, lda_cl$cluster)
plot(prcomp(norm_lda)$x,col=lda_cl$cluster)
#Confusion Matrix
lda_Confm
#SSE
lda_cl$totss
lda_cl$tot.withinss
100-(lda_cl$tot.withinss/lda_cl$totss)*100
#Accuracy
(sum(apply(lda_Confm, 1, max))/sum(lda_cl$size))*100


#LSA/SVD
s<-svd(dtms)
D <- diag(s$d)
D50 <- D[1:50,1:50]
u <- as.matrix(s$u[,1:50])
v <- as.matrix(s$v[,1:50])
d <- as.matrix(diag(s$d)[1:50,1:50])
mat50 <- as.matrix(u%*%d%*%t(v))
mat50
mat50 <- s$u[,1:50] %*% D50 %*% t(s$v[,1:50])

SVD<-function(mat,num){
  sv<-svd(mat,nu = min(nrow(mat),ncol(mat)),nv = min(nrow(mat),ncol(mat)))
  u<-as.matrix(sv$u[,1:num])
  v<-as.matrix(sv$v[,1:num])
  d<-as.matrix(diag(sv$d)[1:num,1:num])
  return(as.matrix(u%*%d%*%t(v),type="blue"))
}

svd_dtm <- SVD(dtms,5)

D100 <- D[1:100,1:100]
D200 <- D[1:200,1:200]

mat100 <- s$u[,1:100] %*% D100 %*% t(s$v[,1:100])
mat200 <- s$u[,1:200] %*% D200 %*% t(s$v[,1:200])
mfreq <- sort(col(s$v),decreasing = TRUE)
S <- svd(dtm)
Dd <- diag( S$d )[ 1:d , 1:d ]
#Mat <- (S%v[,1:d] %*% Dd %*% t(S%u[,1:d])


