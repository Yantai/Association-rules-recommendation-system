#############################################
# association rules and recommendation system
# Yantai Liao
############################################
rm(list=ls())
setwd("/Users/liaoyantai/Documents/study/546 data mining2/homework1")
graphics.off()
library(arules)
load("boston.Rdata")
head(boston)
dim(boston)

#a
quartz()
par(mfrow = c(2,2))
hist(boston$crim)
hist(boston$zn)
hist(boston$indus)
hist(boston$chas)



quartz()
par(mfrow = c(2,2))
hist(boston$nox)
hist(boston$rm)
hist(boston$age)
hist(boston$dis)

quartz()
par(mfrow = c(2,2))
hist(boston$rad)
hist(boston$tax)
hist(boston$ptratio)
hist(boston$black)

quartz()
par(mfrow = c(1,2))
hist(boston$lstat)
hist(boston$medv)


quantile(boston$crim,probs=c(0,0.25,0.70,0.9,1)) # 0.006320  0.082045  1.728440 10.753000 88.976200 
boston[["crim"]]<-ordered(cut(boston[["crim"]],c(0.006,0.083,1.73,10.76,89)),labels=c("Low-crim","Middle-crim","High-crim","Super-crim"))


quantile(boston$zn,probs=c(0,0.75,0.85,0.95,1)) # 0.0  12.5  28.0  80.0 100.0
boston[["zn"]]<-ordered(cut(boston$zn,c(-0.1,12.5,28,80,101)),labels=c("no-lot","low-lot","middle-lot","high-lot"))

quantile(boston$indus,probs=c(0,0.25,0.70,0.9,1)) # 0.46  5.19 18.10 19.58 27.74
boston[["indus"]]<-ordered(cut(boston$indus,c(0.45,5.19,18.1,19.58,28)),labels=c("Low-indus","Middle-indus","High-indus","Super-indus"))

boston[["chas"]]<-ordered(boston$chas,labels=c("off-river","near-river"))

quantile(boston$nox,probs=c(0,0.25,0.70,0.9,1)) # 0.385 0.449 0.605 0.713 0.871
boston[["nox"]]<-ordered(cut(boston$nox,c(0.3,0.449,0.605,0.713,0.9)),labels=c("Low-nox","Middle-nox","High-nox","Super-noxs"))


boston[["rm"]]<-ordered(cut(boston$rm,c(3,5,7,9)),labels=c("small-house","Middle-house","big-house"))

boston[["age"]]<-ordered(cut(boston$age,c(2, 25, 45, 65, 100)),labels=c("Young", "Middle-aged", "Senior", "Elderly"))

boston[["dis"]]<-ordered(cut(boston$dis,c(1, 4, 7, 10, 13)),labels=c("Low-dis","Middle-dis","High-dis","Super-dis"))

boston[["rad"]]<-ordered(cut(boston$rad,c(0.5, 3,5 , 8, 24)),labels=c("Low-index","Middle-index","High-index","Super-index"))

quantile(boston$tax,probs=c(0,0.25,0.70,0.9,1)) # 187  279  437  666  711
boston[["tax"]]<-ordered(cut(boston$tax,c(180,300,500,720)),labels=c("Low-tax","Middle-tax","High-tax"))

quantile(boston$ptratio,probs=c(0,0.25,0.70,0.9,1)) # 12.6 17.4 20.2 20.9 22.0
boston[["ptratio"]]<-ordered(cut(boston$ptratio,c(12,17,20,23)),labels=c("Low-ptratio","Middle-ptratio","High-ptratio"))

quantile(boston$black,probs=c(0,0.25,0.70,0.9,1)) # 12.6 17.4 20.2 20.9 22.0
boston[["black"]]<-ordered(cut(boston$black,c(0,100,300,400)),labels=c("Low-black","Middle-black","High-black"))

quantile(boston$lstat,probs=c(0,0.25,0.70,0.9,1)) #  1.730  6.950 15.620 23.035 37.970 
boston[["lstat"]]<-ordered(cut(boston$lstat,c(1,10,20,40)),labels=c("Low-lstat","Middle-lstat","High-lstat"))

quantile(boston$medv,probs=c(0,0.25,0.70,0.9,1)) #  5.000 17.025 24.150 34.800 50.000 
boston[["medv"]]<-ordered(cut(boston$medv,c(0,15,25,40,55)),labels=c("Low-value","Middle-value","High-value","super-value"))

data<-as(boston,"transactions")
summary(data)

#b
quartz()
itemFrequencyPlot(data,support=0,cex.name= 0.8)

rules<-apriori(data,parameter= list(support = 0.02,confidence = 0.8))
summary(rules)

#c
rulesLowcrim<-subset(rules,subset = lhs %in% "dis=Low-dis" & rhs %in% "crim=Low-crim" & lift>1.2)
rulesLowcrim

inspect(head(sort(rulesLowcrim, by = "confidence"), n = 5))

#d
rulesLowptratio<-subset(rules,subset= rhs %in% "ptratio=Low-ptratio" & lift>1.2)
rulesLowptratio

inspect(head(sort(rulesLowptratio, by="confidence"),n=5))

# regression model

load("boston.Rdata")
boston<-boston[,-14]
set.seed(1)
training<-sample(1:nrow(boston),0.8*nrow(boston))
train<-boston[training,]
test<-boston[-training,]
y.test<-test$ptratio
test<-test[,-11]

train.fit.ols<-lm(ptratio~.,data=train)
y.test.pre.ols<-predict(train.fit.ols,test )
error.ols.test<-mean((y.test.pre.ols-y.test)^2)
error.ols.test # 2.750956

library(glmnet)
x<-as.matrix(train[,-11])
y<-train$ptratio
z<-as.matrix(test)
train.fit.lasso<-glmnet(x,y,alpha=1)
cv.out<-cv.glmnet(x,y,alpha=1)
bestlambda.lasso<-cv.out$lambda.min
bestlambda.lasso #  0.001713947
y.test.pre.lasso<-predict(train.fit.lasso,s=bestlambda.lasso,newx=z,type="response")
error.lasso.test<-mean((y.test.pre.lasso-y.test)^2)
error.lasso.test #2.746526
coef.lasso<-predict(train.fit.lasso,s=bestlambda.lasso,type="coefficients")

r<-names(test)[summary(coef.lasso)$i-1]
c<-summary(coef.lasso)$x[-1]
lasso.model<-data.frame(r,c)
lasso.model

quartz()
barplot(c, width = 1, names.arg =r , beside = T,
        col = c(1:12),  ylab = "value ",ylim=c(-10,10),main="lasso model coefficient")


