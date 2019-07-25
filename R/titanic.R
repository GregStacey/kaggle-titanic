
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



# load and merge iterations
fns = dir("data/data/", full.names = T)
df.stack = merge.titanic(fns)



# make paramid explicitly
sets = paste(df.stack$n.base, df.stack$stack.class,
                df.stack$stack.feat, df.stack$feat.removed, sep="")
unqsets = unique(sets)
df.stack$paramid = match(sets, unqsets)

# make parmodid explicitly
setmods = paste(df.stack$n.base, df.stack$model, df.stack$stack.class,
                df.stack$stack.feat, df.stack$feat.removed, sep="")
unqsetmods = unique(setmods)
df.stack$parmodid = match(setmods, unqsetmods)


# rank the models in each paramid+iter
df.stack$model.rank = numeric(nrow(df.stack))
unqiter = unique(df.stack$iter)
unqparam = unique(df.stack$paramid)
for (ii in 1:length(unqiter)) {
  for (jj in 1:length(unqparam)) {
    I = df.stack$iter==unqiter[ii] & df.stack$paramid==unqparam[jj]
    if (sum(I)==0) next
    tmp = df.stack[I,]
    df.stack$model.rank[I] = order(tmp$Accuracy, decreasing = T)
  }
}
df.stack$model.rank = (df.stack$model.rank-1) / (df.stack$n.base)

# get the average accuracy over iters
unqmods = unique(df.stack$model)
df.stack$avgacc= numeric(nrow(df.stack))
df.stack$nreps= numeric(nrow(df.stack))
for (ii in 1:length(unqparam)) {
  for (jj in 1:length(unqmods)) {
    I = df.stack$paramid==unqparam[ii] & df.stack$model==unqmods[jj]
    df.stack$avgacc[I] = mean(df.stack$Accuracy[I], na.rm=T)
    df.stack$nreps[I] = sum(I)
  }
}

# remove anything with <N min.reps
min.reps = 5
df.stack = df.stack[df.stack$nreps>=min.reps, ]




# data exploration

# when was stacking worth it?
# Answer: all the time (on average)
ggplot(df.stack, aes(y=model.rank, x=model)) + geom_boxplot() + 
  facet_grid(stack.class ~ n.base)
#ggsave("/Users/gregstacey/Professional/kaggle/titanic/figures/model_rank.jpg")

# Q: n.base? stack.class? feat.remove? stack.feat?
# A: four base models, rf, remove Cabin
I = df.stack$model=="stack"
ggplot(df.stack[I,], aes(y=Accuracy, x=feat.removed, color=stack.feat)) + geom_boxplot() + 
  facet_grid(n.base ~ stack.class)




# find the best parameter set
best.parmod.id = df.stack$parmodid[which.max(df.stack$avgacc)]
I.best = which(df.stack$parmodid == best.parmod.id)

# compare it to every other parameter set
allparams = unique(df.stack$parmodid)
pp = data.frame(pp = rep(NA, length(allparams)),
                mag = numeric(length(allparams))) 
for (ii in 1:length(allparams)) {
  I = which(df.stack$parmodid == allparams[ii])
  y = df.stack$Accuracy[I]
  y = y[!is.na(y)]
  if (length(y) < min.reps) next
  
  pp$pp[ii] = wilcox.test(df.stack$Accuracy[I.best], y)$p.value
  pp$mag[ii] = mean(df.stack$Accuracy[I.best] - mean(y))
}



