
# packages
require(tidyr)
require(caret)
require(dplyr)
require(readr)
require(stringr)


# global parameters
classifiers = c("svmLinear", # svm
                "rf", "ranger", "Rborist", # random forest
                "xgbDART","xgbLinear","xgbTree", # xgboost
                "spls", # pls-da
                "glmnet") 
stack.classifier = c("svmLinear", "rf", "glmnet")


# functions
pre.titanic = function(data){
  # feature engineer
  # 1. convert name to [mr, mrs, miss]
  data$title = character(nrow(data))
  data$title[grep("mr.", tolower(data$Name))] = "mr"
  data$title[grep("mrs.", tolower(data$Name))] = "mrs"
  data$title[grep("miss", tolower(data$Name))] = "miss"
  data$title[is.null(data$title) | data$title==""] = "none"
  # 2. n.acq = number of acquaintances = sipsp + parch
  if ("SibSp"%in%names(data) & "Parch"%in%names(data)) {
    data$n.acq = data$SibSp + data$Parch
  }
  # 3. n.cabin = number of cabin assignments
  if ("Cabin"%in%names(data)) {
    data$n.cabin = str_count(data$Cabin, " ") + 1
    data$n.cabin[is.na(data$n.cabin)] = 0
    
    # 4. letter.cabin = letter of cabin assignment (deck?)
    data$letter.cabin = sapply(gsub('[[:digit:]]+', '', data$Cabin),
                               substr, 1, 1)
    data$letter.cabin[is.na(data$letter.cabin) | data$letter.cabin=="T"] = "none"
  }
  
  
  # impute age with regression # DATA LEAKAGE!
  xx.feats = c("Pclass", "title", "Sex", "SibSp", "Parch","Fare",
               "n.acq", "n.cabin", "letter.cabin")
  xx.feats = intersect(xx.feats,names(data))
  yy = data$Age
  xx = data[,xx.feats]
  df = as.data.frame(cbind(yy,xx))
  names(df) = c("Age", names(xx))
  I = is.na(data$Age) # passengers with missing age
  # only use factors with equal levels in both splits
  char.columns = names(select_if(df, is.character))
  bad.columns = character(length(char.columns))
  for (ii in 1:length(char.columns)) {
    if (!identical(unique(df[I,char.columns[ii]]), unique(df[!I,char.columns[ii]]))) {
      bad.columns[ii] = char.columns[ii]
    }
  }
  good.columns = names(df)
  good.columns = good.columns[!good.columns %in% bad.columns]
  fit = glm(df$Age[!I] ~ ., data=df[!I,good.columns])
  data$Age[I] = predict(fit, newdata = df[I,good.columns])
  
  
  # make predictor, response explicitly

  XX = data[,xx.feats]
  YY = factor(data$Survived, levels=c(0,1))
  levels(YY) = c("dead", "alive")
  
  
  # create dummy variables (one-hot encoding?)
  dmy = dummyVars(" ~ .", data = XX, fullRank=F)
  XX = as.data.frame(predict(dmy, newdata = XX))
  
  out = list(XX,YY)
  return(out)
}

create.meta = function(data, mods, include.pred){
  names0 = names(data)
  
  preds.to.include = list()
  if (include.pred) {
    for (ii in 1:length(mods)) {
      preds.to.include[[ii]] = mods[[ii]]$pred
    }
  } else {
    for (ii in 1:length(mods)) {
      preds.to.include[[ii]] = data.frame(dead = rep(NA,nrow(data)), alive = rep(NA,nrow(data)))
    }
  }
  
  # add 1 column types: prob.alive - prob.dead
  col.names = vector(length = length(mods) * 1)
  I.add.col = ncol(data)
  for (ii in 1:length(mods)) {
    I.add.col = I.add.col+1
    data[,I.add.col] = preds.to.include[[ii]]$alive - preds.to.include[[ii]]$dead
    col.names[I.add.col - ncol(XX.train)] = paste(mods[[ii]]$method, "prob.alive", sep=".")
  }
  names(data) = c(names0, col.names)
  
  return(data)
}


merge.titanic = function(fns) {
  
  for (ii in 1:length(fns)) {
    bad.flag = 1
    result = tryCatch({
      load(fns[ii])
      bad.flag = 0
    }, error = function(error_condition) {
      print(ii)
    })
    if (bad.flag==1) next
    if (ii==1) {
      tmp = df.stack
    } else {
      tmp = rbind(tmp, df.stack)
    }
  }
  
  df.stack = tmp
  
  return(df.stack)
}