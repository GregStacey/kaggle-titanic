
# packages
require(tidyr)
require(caret)
require(dplyr)
require(readr)
require(stringr)


# global parameters
classifiers = c("svmRadial", # svm
                "rf", "ranger", # random forest
                "xgbDART","xgbLinear","xgbTree", # xgboost
                "spls", # pls-da
                "glmnet") 
stack.classifier = c("svmRadial", "rf", "glmnet")


# functions
pre.titanic = function(data){
  # feature engineer
  remove.feats = c("PassengerId", "Survived", "Cabin", "Name", "Ticket")
  # 1. PassengerId - REMOVE
  # 2. Survived - REMOVE (obvi)
  # 3. Pclass - keep
  # 4. Name - REMOVE, convert to $title = [mr, mrs, miss, master]
  data$title = character(nrow(data))
  data$title[grep("mr.", tolower(data$Name))] = "mr"
  data$title[grep("mrs.", tolower(data$Name))] = "mrs"
  data$title[grep("miss", tolower(data$Name))] = "miss"
  data$title[grep("master", tolower(data$Name))] = "master"
  data$title[is.null(data$title) | data$title==""] = "none"
  # 5. Sex - keep
  # 6. Age - keep + impute missing
  # 7. SipSp - keep
  # 8. Parch - keep
  #          - convert to $n.acq = number of acquaintances = sipsp + parch
  #          - convert to $is.alone = n.acq == 0
  data$n.acq = data$SibSp + data$Parch
  data$is.alone = as.numeric(data$n.acq == 0)
  # 9. Ticket - REMOVE, convert to $ticket.type = [none, rare]
  # tmp = gsub('[[:digit:]]+.', '', data$Ticket) # remove number
  # tmp = gsub("3", "", tmp)
  # tmp = unique(tolower(unlist(sapply(tmp, function(x) unique(strsplit(x, "")[[1]])))))
  # tmp = tmp[!tmp==" "]
  # tmp[tmp=="."] = "[.]"
  # # only keep those with >20 occurrences
  # nn = numeric(length(tmp))
  # for (ii in 1:length(tmp)) {
  #   nn[ii] = sum(grepl(tmp[ii], tolower(data$Ticket)))
  # }
  # tmp = tmp[nn>=20]
  tmp = c( "a","/","p","c","s","t","o","n","[.]")
  # add as one-hot-encoding columns
  for (ii in 1:length(tmp)) {
    data[,paste("ticket.type.", tmp[ii], sep="")] = as.numeric(grepl(tmp[ii], tolower(data$Ticket)))
  }
  # 10. Cabin - REMOVE
  # 11. Embarked - keep + transform to Embarked = [S, C, Q]
  data$Embarked[is.na(data$Embarked)] = "S"
  
  
  # impute age with regression # DATA LEAKAGE!
  xx.feats = names(data)[!names(data) %in% c(remove.feats, "Age")]
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
  XX = data[,! names(data) %in% remove.feats]
  YY = factor(data$Survived, levels=c(0,1))
  levels(YY) = c("dead", "alive")
  
  
  # create dummy variables (one-hot encoding?)
  dmy = dummyVars(" ~ .", data = XX, fullRank=T)
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
    print(I.add.col - length(names0))
    col.names[I.add.col - length(names0)] = paste(mods[[ii]]$method, "prob.alive", sep=".")
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


feature.power = function(XX, yy) {
  if (is.factor(yy)) {
    yy = match(yy, levels(yy))
  } else if (is.character(yy)) {
    yy = match(yy, sort(unique(yy)))
  }
  
  ds = data.frame(feature = names(XX),
                  strength = numeric(ncol(XX)),
                  zscore = numeric(ncol(XX)), stringsAsFactors = F)
  for (ii in 1:ncol(XX)) {
    x = aggregate(yy ~ XX[,ii], data = XX, FUN = mean)
    ds$strength[ii] = lm(x[,2] ~ x[,1])$coefficients[2]
    zz = numeric(100)
    for (jj in 1:100) {
      x = aggregate(sample(yy, length(yy)) ~ XX[,ii], data = XX, FUN = mean)
      zz[jj] = abs(lm(x[,2] ~ x[,1])$coefficients[2])
    }
    ds$zscore[ii] = (abs(ds$strength[ii]) - mean(zz)) / sd(zz)
  }
  
  return(ds)
}


train.stack = function(data.train, data.test, params) {
  hi = 1
  
  print(1)

  # pre-process (feature engineer, etc.)
  tmp = pre.titanic(data.train)
  XX.train = tmp[[1]]
  YY.train = tmp[[2]]
  tmp = pre.titanic(data.test)
  XX.test = tmp[[1]]
  YY.test = tmp[[2]]
  
  # cleaning: remove any features not in common
  badfeats = setdiff(names(XX.train), names(XX.test))
  XX.train = XX.train[,!names(XX.train) %in% badfeats]
  XX.test = XX.test[,!names(XX.test) %in% badfeats]
  
  print(2)
  
  # I. filter BASE features by discriminating power
  ds = feature.power(XX.train, YY.train)
  good.feats = ds$feature[ds$zscore > params$base.feature.strength[hi]]
  XX.train = XX.train[, names(XX.train) %in% good.feats]
  XX.test = XX.test[, names(XX.test) %in% good.feats]
  

  ### 1. Divide data into folds
  nfold = 5
  ndata = nrow(XX.train)
  foldID = sample(ceiling(seq(from=1/ndata, to=1, length=ndata)*nfold), ndata)
  # make ctrl.index for trainControl
  ctrl.index = list()
  for (ii in 1:nfold) {
    ctrl.index[[ii]] = which(!foldID==ii) # rows NOT in this fold
  }

  # confirm no folds-minus-one have zero variance
  # if there are, remove them
  flag.bad = numeric(ncol(XX.train))
  for (ii in 1:nfold) {
    for (jj in 1:ncol(XX.train)) {
      tmp.var = var(XX.train[ctrl.index[[ii]],jj])
      if (is.na(tmp.var) | is.infinite(tmp.var)) next
      if (tmp.var==0) flag.bad[jj] = 1
    }
  }
  featsbad = character(0)
  if (sum(flag.bad)>0) {
    featsbad = names(XX.train)[which(flag.bad>0)]
  }
  XX.train = XX.train[,!names(XX.train) %in% featsbad]
  XX.test = XX.test[,!names(XX.test) %in% featsbad]
  
  
  print(3)
  

  ### 2. For base models M, hyperparm optimize using train.
  ctrl = trainControl(method = "cv", number=5, classProbs = T, index=ctrl.index)
  mods = list()
  for (ii in 1:length(classifiers)) {
    print(classifiers[ii])
    mods[[ii]] = caret::train(x=XX.train, y=YY.train,
                              method = classifiers[ii], 
                              trControl = ctrl)
    mods[[ii]]$pred = predict(mods[[ii]], type='prob') # add 'pred', since it's missing for some reason!
  }

  # select some subset of mods...
  # II) take the nn.mods best base models
  nn.mods = min(length(mods), params$base.range[hi])
  mod.acc = numeric(length(mods))
  for (ii in 1:length(mods)) {
    mod.acc[ii] = max(mods[[ii]]$results$Accuracy)
  }
  I = order(mod.acc, decreasing = T)
  classifiers = classifiers[I[1:nn.mods]]
  mods = mods[I[1:nn.mods]]
  
  
  print(4)
  
  
  ### 3. Create train_meta and test_meta.
  ### 4. Using folds, fit each M to train and predict on the holdout fold.
  #      Store predictions in train_meta.
  meta.train = create.meta(XX.train, mods, T)
  meta.test = create.meta(XX.test, mods, F)
  
  # III. filter STACK features by discriminating power
  if (params$stack.feature.strength[hi] == "just.cass") {
    meta.train = meta.train[,grepl("prob.alive", names(meta.train))]
    meta.test = meta.test[,grepl("prob.alive", names(meta.test))]
  } else {
    ds = feature.power(meta.train, YY.train)
    good.feats = ds$feature[ds$zscore > as.numeric(params$stack.feature.strength[hi])]
    meta.train = meta.train[, names(meta.train) %in% good.feats]
    meta.test = meta.test[, names(meta.test) %in% good.feats]
  }
  
  
  ### 5. Fit each M to all of train, predict on test.
  #       Store predictions in test_meta.
  mods.whole = list()
  ctrl.whole = trainControl(method = "none", classProbs = TRUE)
  for (ii in 1:length(classifiers)) {
    print(classifiers[ii])
    # fit M to train
    mods.whole[[ii]] = caret::train(x=XX.train, y=YY.train,
                                    method = classifiers[ii], 
                                    tuneGrid = mods[[ii]]$bestTune, # pass opt hyperparams
                                    trControl = ctrl.whole)
    # predict on test
    aa = predict(mods.whole[[ii]], XX.test, type="prob")
    # store in test_meta
    col.name = paste(classifiers[ii], "prob.alive", sep=".")
    meta.test[,col.name] = aa$alive - aa$dead
  }
  

  ### 6. Fit the stacking model S to train_meta, predict on test_meta.
  # fit S to train_meta
  # iii) change stacking model
  mod.final = caret::train(x=meta.train, y=YY.train,
                           method = params$stack.class[hi],
                           trControl = ctrl.whole)
  

  # report final accuracies - single models and stacked
  # Stack
  df.stack = data.frame(Accuracy = numeric(10^2),
                        model = character(10^2), 
                        iter = numeric(10^2), 
                        base.feature.strength = numeric(10^2), 
                        base.range = numeric(10^2), 
                        stack.feature.strength = character(10^2), 
                        stack.classifier = character(10^2), 
                        paramid = numeric(10^2), stringsAsFactors = F)
  cc = 1
  predictions.final = predict(mod.final, meta.test, type="raw")
  df.stack$Accuracy[cc] = sum(predictions.final == YY.test) / length(YY.test)
  df.stack$model[cc] = "stack"
  df.stack$iter[cc] = params$iter[hi]
  df.stack$base.feature.strength[cc] = params$base.feature.strength[hi]
  df.stack$base.range[cc] = params$base.range[hi]
  df.stack$stack.feature.strength[cc] = params$stack.feature.strength[hi]
  df.stack$stack.classifier[cc] = params$stack.classifier[hi]
  df.stack$paramid[cc] = hi
  for (ii in 1:length(classifiers)) {
    cc = cc+1
    predictions.final = predict(mods.whole[[ii]], XX.test, type="raw")
    df.stack$Accuracy[cc] = sum(predictions.final == YY.test) / length(YY.test)
    df.stack$model[cc] = classifiers[ii]
    df.stack$iter[cc] = params$iter[hi]
    df.stack$base.feature.strength[cc] = params$base.feature.strength[hi]
    df.stack$base.range[cc] = params$base.range[hi]
    df.stack$stack.feature.strength[cc] = params$stack.feature.strength[hi]
    df.stack$stack.classifier[cc] = params$stack.classifier[hi]
    df.stack$paramid[cc] = hi
  }
  print(10)
  df.stack = df.stack[1:cc,]
  
  out = list(df.stack, mod.final, meta.train, meta.test, XX.train, XX.test)
  
  return(out)
}