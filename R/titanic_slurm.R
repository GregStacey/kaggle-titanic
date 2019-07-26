
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
# 4. stacking features
source("R/functions.R")
base.feature.strength = c(0, 2, 4) # what goes into base models
base.range = c(4, 9) # what comes out of base models
stack.feature.strength = c(0, 10, "just.class") # what goes into stack models
# stack.classifier # stacking algorithm
df.xval.control = data.frame(iter = numeric(10^4), 
                             base.feature.strength = numeric(10^4),
                             base.range = numeric(10^4),
                             stack.feature.strength = numeric(10^4),
                             stack.classifier = character(10^4),
                             paramid = numeric(10^4), stringsAsFactors = F)
cc = 0
id = 0
iterMax = 18
for (ii in 1:length(base.feature.strength)) {
  for (jj in 1:length(base.range)) {
    for (mm in 1:length(stack.feature.strength)) {
      for (kk in 1:length(stack.classifier)) {
        id = id+1
        for (iter in 1:iterMax) {
          cc = cc+1
          df.xval.control$iter[cc] = iter
          df.xval.control$base.feature.strength[cc] = base.feature.strength[ii]
          df.xval.control$base.range[cc] = base.range[jj]
          df.xval.control$stack.feature.strength[cc] = stack.feature.strength[mm]
          df.xval.control$stack.classifier[cc] = stack.classifier[kk]
          df.xval.control$paramid[cc] = id
        }
      }
      
    }
  }
}
df.xval.control = df.xval.control[1:cc,]



# read command line argument
# it tells us where we are in df.xval.control
hi = as.numeric(commandArgs(trailingOnly = T))
print(paste("command arg is", hi))


source("R/functions.R")

# load data
data.train = as.data.frame(read_csv("./data/train.csv"))
# for optimizing, split data.train (w/ labels) into train and test
I.test = (1:nrow(data.train)) %in% sample(nrow(data.train), round(nrow(data.train) / 5))
data.test = data.train[I.test,]
data.train = data.train[!I.test,]


# train and predict stacking classifier
tmp = train.stack(data.train, data.test, df.xval.control[hi,])
df.stack = tmp[[1]]
# save
save(df.xval.control, df.stack, 
     file=paste("./data/data3/stack.accuracy_", hi, ".Rda", sep=""))


