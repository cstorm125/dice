les_packages <- lapply(c('xgboost','Matrix','rBayesianOptimization','caret','ade4','tidyverse',
                         'rpart','rpart.plot','ggthemes','DT','reshape2','dplyr'),
                       require, character.only = TRUE)

input_path <- '../files/'

fan_data = fread(paste0(input_path,'fan_data.csv')) %>% data.frame()

#multiple choices
all_stuff = list(c('math','science','language','social','pe','others'),
                 c('action','drama','horror','comedy','fantasy','scifi','romance','others'),
                 c('LINE','Facebook','Twitter','Instagram','WhatsApp','WeChat','BeeTalk','others'),
                 c('sports','games','video','friends','reading','club','others'))
names(all_stuff) = c('sub','mov','sns','hob')

mult = data.frame(row.names = 1:70)

for(j in names(all_stuff) ){
  for(i in all_stuff[[j]]){
    mult[,paste0(j,'_',i)] = sapply(fan_data[,j],function(x){
      ifelse(grepl(i,x),1,0)
    })
  }
}

#categorical
cat = fan_data %>% select(slp,onphone,nb_freinds,nb_diff,gender,order)
cat = apply(cat,1,as.factor) %>% t() %>% data.frame()
cat_ohe = acm.disjonctif(cat)

#numerical
num = fan_data %>% select(age,weight,height,gpa,past,siblings,
                          physical,attract,social,reason,judge,
                          evidence)

#master dataframe
survey_sample = data.frame(y=fan_data$target,num,cat_ohe,mult)
saveRDS(survey_sample,paste0(input_path,'survey_sample.rds'))

survey_sample = readRDS(paste0(input_path,'survey_sample.rds'))

#exploration

#categorical
#cat
cat_df = NULL
for(i in names(cat)){
  cat_df = rbind(cat_df,table(cat[,i]) %>% data.frame %>% mutate(col=i))
}
names(cat_df) = c('variable','value','col')

#mult
mult_agg = colSums(mult) %>% t() %>% data.frame
sub_df = mult_agg[,grepl('sub',colnames(mult_agg))] %>% melt %>% mutate(col='sub')
mov_df = mult_agg[,grepl('mov',colnames(mult_agg))] %>% melt %>% mutate(col='mov')
sns_df = mult_agg[,grepl('sns',colnames(mult_agg))] %>% melt %>% mutate(col='sns')
hob_df = mult_agg[,grepl('hob',colnames(mult_agg))] %>% melt %>% mutate(col='hob')
mult_df = rbind(sub_df,mov_df,sns_df,hob_df) %>%
  mutate(variable = sapply(variable,function(x) strsplit(as.character(x),'_')[[1]][2]))

mult_df = rbind(mult_df,cat_df)
mult_df_total = mult_df %>% group_by(col) %>% summarise(total=sum(value))
mult_df = mult_df %>% inner_join(mult_df_total) %>% mutate(per=value/total)

saveRDS(mult_df,paste0(input_path,'mult_df.rds'))

g = ggplot(mult_df,aes(x=variable,y=per,fill=col)) + geom_bar(stat='identity') +
  facet_wrap(~col,ncol = 5,scale='free') + theme_fivethirtyeight() + 
  scale_y_continuous(labels = scales::percent) +
  theme(legend.position = 'none',axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle('% Response for Each Categorical Variables')
g

#numeric descriptive
descript <- num
summary(descript)

descript$id_var <- 1
descript_m <- melt(descript,id='id_var')
descript_m <- subset(descript_m,select=-c(id_var))
descript_m$value <- as.numeric(descript_m$value )

saveRDS(descript_m,paste0(input_path,'descript_m.rds'))

g<-ggplot(descript_m,aes(x=value,fill=variable)) + geom_histogram() + 
  facet_wrap(~variable,scales=c('free')) + theme_fivethirtyeight() +
  theme(legend.position = 'none') + ggtitle('Distribution of Numerical Variables')
g

#correlation
M <- cor(survey_sample)
d3heatmap(M, scale='column') 

#modeling

#train-valid
set.seed(1412)
survey_sample = readRDS(paste0(input_path,'survey_sample.rds'))
#selected='y|hob|sub|sns|past|gpa|weight|height|age|attract|social|physical|onphone'
#survey_sample = survey_sample[,grepl(selected,names(survey_sample))]

inTrain = KFold(survey_sample$y,nfold = 10, stratified = TRUE, seed = 1412)
train_set = survey_sample[unlist(inTrain[1:7]),]
test_set = survey_sample[unlist(inTrain[8:10]),]

dtrain <- xgb.DMatrix(data.matrix(train_set[,2:dim(train_set)[2]]), label = train_set$y)
dtest <- xgb.DMatrix(data.matrix(test_set[,2:dim(train_set)[2]]), label = test_set$y)
watchlist = list(eval=dtest, train = dtrain)

#fitting
params = list(eval_metric = 'auc',nthread = 4, objective = "binary:logistic",
                 booster = 'gbtree', verbose = 0)

fit = xgb.train(params=params,data = dtrain, nrounds = 20, watchlist=watchlist, silent = 1)

#rpart
fit_rpart = rpart(y~.,data = train_set)
saveRDS(fit_rpart,paste0(input_path,'fit_rpart.rds'))
rpart.plot(fit_rpart)

#feature importance
importance = xgb.importance(feature_names = names(train_set)[2:dim(train_set)[2]], model = fit)
importance[,2:4] = round(100*importance[,2:4],2)
importance$sign = sapply(importance$Feature, FUN= function(x){
  ifelse(cor(as.numeric(train_set$y),as.numeric(train_set[,x])) > 0, 'plus', 'minus')
})

saveRDS(importance,paste0(input_path,'importance.rds'))
g = ggplot(importance[1:10,], aes(x=reorder(Feature,-Gain),y=Gain/100,fill=sign)) + geom_bar(stat='identity') +
  scale_y_continuous(labels = scales::percent) + theme_fivethirtyeight() + scale_fill_wsj() +
  ggtitle('Top 10 Most Important Variables') + coord_flip() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
g

#predict
pred = predict(fit,dtest)
pred_lab = ifelse(pred>0.5,1,0)
cm = confusionMatrix(pred_lab,test_set$y,positive = '1')

saveRDS(cm,paste0(input_path,'cm.rds'))

#https://stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package
draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('Confusion Matrix', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'No', cex=2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Yes', cex=2)
  text(125, 370, 'Predicted', cex=2, srt=90, font=2)
  text(245, 450, 'Actual', cex=2, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'No', cex=2, srt=90)
  text(140, 335, 'Yes', cex=2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=2, font=2, col='white')
  text(195, 335, res[2], cex=2, font=2, col='white')
  text(295, 400, res[3], cex=2, font=2, col='white')
  text(295, 335, res[4], cex=2, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=2, font=2)
  text(30, 85, names(cm$byClass[2]), cex=2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=2, font=2)
  text(50, 85, names(cm$byClass[5]), cex=2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=2, font=2)
  text(70, 85, names(cm$byClass[6]), cex=2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=2, font=2)
  text(90, 85, names(cm$byClass[7]), cex=2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=2, font=2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=2, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=2)
  text(70, 35, names(cm$overall[2]), cex=2, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=2)
}  

draw_confusion_matrix(cm)
