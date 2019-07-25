
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






# oops! I got paramid wrong
sets = paste(df.stack$n.base, df.stack$stack.class, 
                df.stack$stack.feat, df.stack$feat.removed, sep="")
unqsets = unique(sets)
df.stack$paramid = match(sets, unqsets)

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
    df.stack$avgacc[I] = mean(df.stack$Accuracy[I])
    df.stack$nreps[I] = sum(I)
  }
}

# final save
df.stack = df.stack[1:cc,]
save(df.xval.control, df.stack, file="./data/stack.accuracy2.Rda")



load("/Users/gregstacey/Professional/kaggle/titanic/data/stack.accuracy2.Rda")
df.stack = df.stack[!df.stack$iter==0,]

ggplot(df.stack, aes(x=model, y=Accuracy)) + geom_boxplot()
ggplot(df.stack, aes(x=paramid, y=Accuracy, color=is.stack)) + geom_point(alpha=.4)
ggplot(df.stack, aes(x=Accuracy, fill=is.stack)) + geom_density(alpha=.4)

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


