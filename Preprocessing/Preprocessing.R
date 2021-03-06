
setwd("D:/AIproject")

# no date data
pjdata<-read.csv("pjdata_pre_cg2.csv")

head(pjdata)
dim(pjdata)
str(pjdata)

attach(pjdata)

quantile(Pat)
quantile(PM10)
quantile(PM2.5)
quantile(Tem)
quantile(Wind)
quantile(Hum)
quantile(Pres)
quantile(NO2)
quantile(O3)
quantile(CO)
quantile(SO2)
anova(PM10, S)
help("anova")

library(car)

summary(m1)

m1<-lm(Pat~., data = pjdata)
vif(m1)
vif(m1)>10

cor(pjdata[0:11])
plot(pjdata[0:11])

# pca
pj4.pca<-prcomp(pjdata[,0:10],center=T,scale.=T)
pj4.pca

summary(pj4.pca)
plot(pj4.pca,type="l")

PRC4<-as.matrix(pjdata[,0:10])%*%pj4.pca$rotation
head(PRC4)
pj4.pc<-cbind(as.data.frame(PRC4),Pat)
fit2<-lm(Pat~PC1+PC2+PC3+PC4+PC5+PC6+PC8, data=pj4.pc)
fit2

summary(fit2)


# randomforest
set.seed(1000)
N<-nrow(pjdata)
tr.idx<-sample(1:N, size=N*2/3, replace=FALSE)
train<-pjdata[tr.idx,]
test<-pjdata[-tr.idx,]


library(randomForest)
help("randomForest")
rf_out2<-randomForest(as.factor(Pat)~.,data=train, importance=T, mtry=2)
rf_out2
round(importance(rf_out2), 2)
varImpPlot(rf_out2)

library(caret)
rfpred<-predict(rf_out2,test)
rfpred

library (e1071)
confusionMatrix(as.factor(rfpred),as.factor(test$Pat))


#svm
set.seed(1000)
N<-nrow(pjdata)
tr.idx<-sample(1:N, size=N*2/3, replace=FALSE)
train<-pjdata[tr.idx,]
test<-pjdata[-tr.idx,]

m1<-svm(as.factor(Pat)~., data = train, kernel = "linear")
summary(m1)
m2<-svm(as.factor(Pat)~., data = train,kernel="polynomial")
summary(m2)
m3<-svm(as.factor(Pat)~., data = train,kernel="sigmoid")
summary(m3)
m4<-svm(as.factor(Pat)~., data = train,kernel="radial")
summary(m4)

pred1<-predict(m1,test)
confusionMatrix(pred1,as.factor(test$Pat))

pred2<-predict(m2,test)
confusionMatrix(pred2,as.factor(test$Pat))

pred3<-predict(m3,test)
confusionMatrix(pred3,as.factor(test$Pat))

pred4<-predict(m4,test)
confusionMatrix(pred4,as.factor(test$Pat))

# lda
library(lda)
library(gmodels)
library(biotools)
library(klaR)
pj.lda <- lda(Pat ~ ., data=train, prior=c(1/3,1/3,1/3))
pj.lda

testpredq <- predict(pj.lda, test)
testpredq

testLabels<-pjdata[-tr.idx,11]
CrossTable(x=testLabels,y=testpredq$class, prop.chisq=FALSE)

pj.qda <- qda(Pat ~ ., data=train, prior=c(1/3,1/3,1/3))
pj.qda

testpredq <- predict(pj.qda, test)
testpredq

library(gmodels)
CrossTable(x=testLabels,y=testpredq$class, prop.chisq=FALSE)


m1<-glm(Pat~., data = train, family = "binomial")
m1
summary(m1)
install.packages("pscl")
library(pscl)
pR2(m1)

install.packages("ROCR")
library(ROCR)
p <- predict(m1, newdata=test, type="response")
pr <- prediction(p, test$Pat)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc