
# follow stacking procedure here:
# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
# 1. Partition train into CV folds
# 2. For base models M, hyperparm optimize using train.
# 3. Create train_meta and test_meta.
# 4. Using folds, fit each M to train and predict on the holdout fold.
#    Store predictions in train_meta.
# 5. Fit each M to all of train, predict on test.
#    Store predictions in test_meta.
# 6. Fit the stacking model S to train_meta, predict on test_meta.





# SET UP STACKING HYPERPARAMETERS
# things to play with
# 1. engineered features. e.g. don't keep cabin numbers
# 2. base models. e.g. only stack the best 5 models
# 3. stacking model.
source("R/functions.R")
base.range = c(9, 4)
feat.removed = c("none", "Ticket", "Cabin", "Embarked")
df.xval.control = data.frame(base = numeric(10^3),
                             stack.class = character(10^3),
                             feat.removed = character(10^3), stringsAsFactors = F)
cc = 0
for (ii in 1:length(base.range)) {
  for (jj in 1:length(stack.classifier)) {
    for (kk in 1:length(feat.removed)) {
      cc = cc+1
      df.xval.control$base[cc] = base.range[ii]
      df.xval.control$stack.class[cc] = stack.classifier[jj]
      df.xval.control$feat.removed[cc] = feat.removed[kk]
    }
  }
}
df.xval.control = df.xval.control[1:cc,]




# Stack
df.stack = data.frame(Accuracy = numeric(10^3),
                      model = character(10^3), 
                      iter = numeric(10^3), stringsAsFactors = F)
cc = 0
iterMax = 10
for (iter in 2:iterMax) {
  for (hi in 1:nrow(df.xval.control)) {
    
    source("R/functions.R")
    print(1)
    # load data
    data.train = as.data.frame(read_csv("./data/train.csv"))
    if (T){
      # for optimizing, split data.train (w/ labels) into train and test
      I.test = (1:nrow(data.train)) %in% sample(nrow(data.train), round(nrow(data.train) / 5))
      data.test = data.train[I.test,]
      data.train = data.train[!I.test,]
    } else {
      # use the real (unlabeled!) test data for testing
      data.test = as.data.frame(read_csv("./data/test.csv"))
    }
    
    # i) remove a feature
    data.train = data.train[,!names(data.train) == df.xval.control$feat.removed[hi]]
    data.test = data.test[,!names(data.test) == df.xval.control$feat.removed[hi]]
    
    print(2)
    
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
    
    print(3)
    
    ### 1. Divide data into folds
    nfold = 5
    ndata = nrow(XX.train)
    foldID = sample(ceiling(seq(from=1/ndata, to=1, length=ndata)*nfold), ndata)
    # make ctrl.index for trainControl
    ctrl.index = list()
    for (ii in 1:nfold) {
      ctrl.index[[ii]] = which(!foldID==ii) # rows NOT in this fold
    }
    print(4)
    
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
    
    print(5)
    
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
    print("5b")
    
    # select some subset of mods...
    # i) take the N best base models
    mod.acc = numeric(length(mods))
    for (ii in 1:length(mods)) {
      mod.acc[ii] = max(mods[[ii]]$results$Accuracy)
    }
    I = order(mod.acc, decreasing = T)
    classifiers = classifiers[I[1:df.xval.control$base[hi]]]
    mods = mods[I[1:df.xval.control$base[hi]]]
    
    
    print(6)
    
    ### 3. Create train_meta and test_meta.
    ### 4. Using folds, fit each M to train and predict on the holdout fold.
    #      Store predictions in train_meta.
    meta.train = create.meta(XX.train, mods, T)
    meta.test = create.meta(XX.test, mods, F)
    
    print(7)
    
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
    
    print(8)
    
    ### 6. Fit the stacking model S to train_meta, predict on test_meta.
    # fit S to train_meta
    # iii) change stacking model
    mod.final = caret::train(x=meta.train, y=YY.train,
                             method = df.xval.control$stack.class[hi],
                             trControl = ctrl.whole)
    
    print(9)
    
    # report final accuracies - single models and stacked
    cc = cc+1
    predictions.final = predict(mod.final, meta.test, type="raw")
    df.stack$Accuracy[cc] = sum(predictions.final == YY.test) / length(YY.test)
    df.stack$model[cc] = "stack"
    df.stack$iter[cc] = iter
    df.stack$paramid[cc] = hi
    for (ii in 1:length(classifiers)) {
      cc = cc+1
      predictions.final = predict(mods.whole[[ii]], meta.test, type="raw")
      df.stack$Accuracy[cc] = sum(predictions.final == YY.test) / length(YY.test)
      df.stack$model[cc] = classifiers[ii]
      df.stack$iter[cc] = iter
      df.stack$paramid[cc] = hi
    }
    print(10)
    
    # save in case of crash
    save(df.xval.control, df.stack, file="./data/stack.accuracy.Rda")
  }
}

# final save
df.stack = df.stack[1:cc,]
save(df.xval.control, df.stack, file="./data/stack.accuracy.Rda")



load("data/stack.accuracy.Rda")
df.stack$paramid[seq(from=1, to=nrow(df.stack), by=10)] = df.stack$paramid[seq(from=1, to=nrow(df.stack), by=10)+1]
df.stack$iter[seq(from=1, to=nrow(df.stack), by=10)] = df.stack$iter[seq(from=1, to=nrow(df.stack), by=10)+1]
ggplot(df.stack[1:cc,], aes(x=model, y=Accuracy)) + geom_boxplot()

