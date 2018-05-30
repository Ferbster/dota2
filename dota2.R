### load libraries
library(readr)
library(tidyverse)
library(jsonlite)
library(caret)


### load data

dota2Train <- read_csv("dota2Train.csv", col_names = FALSE)

dota2Train =
dota2Train %>%
  mutate(ID = sample(1000000 : 9999999, nrow(.), replace = F)) %>% 
  select(ID, 1:117)


hero_json = "https://raw.githubusercontent.com/kronusme/dota2-api/master/data/heroes.json"
hero_names = fromJSON(hero_json)$heroes %>% 
  arrange(id)

### adding names from Json + missing monkey_king
colnames(dota2Train) = c("ID", "result", "cluster_id", "game_mode", "game_type", hero_names$name, "monkey_king")


hero_stats =
  dota2Train %>% 
  gather(hero, team, -c(1:5))

###### Feature creation ####
### Unit of analsis: Hero ####

### Individual hero winning percentage

hero_summary =
  hero_stats %>% 
  filter(team != 0) %>% 
  mutate(win = if_else(result == team, 1,0)) %>% 
  group_by(hero) %>% 
  summarise(win_perc = sum(win)/n()) 



### Co-occurence matrix for Team 1

heroes_coop_1 =
  hero_stats %>% 
  mutate(team = if_else(team == 1, 1,0)) %>% 
  select(1, 6,7) %>% 
  spread(hero, team) %>% 
  select(2:114) %>% 
  as.matrix(.) %>% 
  crossprod(.)  



### Co-occurence matrix for Team -1

heroes_coop_n1 =
  hero_stats %>% 
  mutate(team = if_else(team == -1, 1,0)) %>% 
  select(1, 6,7) %>% 
  spread(hero, team) %>% 
  select(2:114) %>% 
  as.matrix(.) %>% 
  crossprod(.)  



heroes_coop = heroes_coop_1 + heroes_coop_n1


### Winning Coop Matrix

heroes_win_matrix =
  hero_stats %>%
  mutate(win = if_else(result == team, 1,0)) %>% 
  select(1, 6,8) %>% 
  spread(hero, win) %>% 
  select(2:114) %>% 
  as.matrix(.)
  
heroes_coop_win = crossprod(heroes_win_matrix)

### Winning Percent matrix

heroes_coop_win_perc = heroes_coop_win/heroes_coop
diag(heroes_coop_win_perc) = 0

### convert to tibble

heroes_dyad_win = as_tibble(heroes_coop_win_perc, rownames = "hero")
colnames(heroes_dyad_win)[-1] = paste0("coop_", colnames(heroes_dyad_win)[-1])


#### Win prob against hero


#### playing against

heroes_against_1 =
  hero_stats %>%
  mutate(team = if_else(team == -1, 1, 0)) %>% 
  select(1, 6,7) %>% 
  spread(hero, team) %>% 
  select(2:114) %>% 
  as.matrix(.)  




heroes_against_n1 =
  hero_stats %>% 
  mutate(team = if_else(team == 1, 1, 0)) %>% 
  select(1, 6,7) %>% 
  spread(hero, team) %>% 
  select(2:114) %>% 
  as.matrix(.)  


heroes_against_matrix = crossprod(heroes_against_1, heroes_against_n1) + crossprod(heroes_against_n1, heroes_against_1)


#heroes_coop_loosing =
heroes_loss_matrix = hero_stats %>% 
  mutate(win = case_when(result == team ~ 0,
                         team != 0 & result != team ~ 1,
                         TRUE ~ 0) ) %>% 
  select(1, 6,8) %>% 
  spread(hero, win) %>% 
  select(2:114) %>% 
  as.matrix(.)
  
heroes_win_against = t(heroes_win_matrix) %*% heroes_loss_matrix

heroes_win_against_perc = heroes_win_against/heroes_against_matrix

### convert to tibble

heroes_vs_hero = as_tibble(heroes_win_against_perc, rownames = "hero")

colnames(heroes_vs_hero)[-1] = paste0("vs_", colnames(heroes_vs_hero)[-1])

### Function to recode Nan to NA
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

heroes_vs_hero[is.nan(heroes_vs_hero)] = 0


##### Combine data to hero feature data frame ####

hero_features = hero_summary %>% 
  inner_join(heroes_dyad_win) %>% 
  inner_join(heroes_vs_hero) 

hero_features[hero_features == 0] = NA
  
##### Build prediction data frame ####
#### Unit of analysis: Game ID ###

hero_stats = hero_stats %>% 
filter(team != 0)

dota_pred_test =
 hero_stats %>% 
  mutate(win = if_else(result == team, 1,0)) %>% 
  left_join(hero_features) %>% 
  group_by(ID, team) %>% 
  gather(co_vs, perc,-c(1:9)) %>% 
  filter(co_vs %in% paste0("coop_", hero) | co_vs %in% paste0("vs_", hero))
  
dota_pred_test =  
  dota_pred_test %>%
  na.omit(dota_pred_test) %>%
  group_by(ID, hero, cluster_id, game_mode, game_type) %>%
  separate(co_vs, "coop_vs", sep = "_") %>%
  summarise(
    team = first(team),
    result = first(result),
    hero_win = first(win_perc),
    coop = mean(perc[coop_vs == "coop"], na.rm = T),
    max_coop = max(perc[coop_vs == "coop"], na.rm = T),
    range_coop = max_coop - min(perc[coop_vs == "coop"], na.rm = T),
    vs = mean(perc[coop_vs == "vs"], na.rm = T),
    max_vs =  max(perc[coop_vs == "vs"], na.rm = T),
    range_vs = max_vs - min(perc[coop_vs == "vs"], na.rm = T)
  )


dota_pred_test = 
dota_pred_test %>% 
  group_by(ID, team) %>% 
  arrange(hero_win) %>% 
  mutate(hero_no = paste0("hero_", team, "_",row_number())) %>%
  gather("metric", "value", -c(1:7, 15)) %>% 
  unite(hero_no, metric, col = "hero_feature", sep = "_") %>% 
  group_by(ID) %>% 
  select(-2, -team) %>% 
  spread(hero_feature, value)
  

#### Model Training #####

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

##### K-nearest neighbours

knn_fit <- train(as.factor(result) ~ ., 
                 data = dota_pred_test, 
                 method = "knn",
                 trControl = fitControl)



##### Ranger Random Forest

ranger_fit <- train(as.factor(result) ~ ., 
                data = dota_pred_test, 
                method = "ranger",
                trControl = fitControl)
##### XGBoost

xgb_fit <- train(as.factor(result) ~ ., 
                    data = dota_pred_test, 
                    method = "xgbTree",
                 trControl = fitControl)



##### work with test data ####
#### I know the code is redundant, but anyway ###


dota2Test <- read_csv("dota2Test.csv", col_names = FALSE)

dota2Test =
  dota2Test %>%
  mutate(ID = sample(1000000 : 9999999, nrow(.), replace = F)) %>% 
  select(ID, 1:117)



### adding names from Json + missing monkey_king
colnames(dota2Test) = c("ID", "result", "cluster_id", "game_mode", "game_type", hero_names$name, "monkey_king")


hero_stats =
  dota2Test %>% 
  gather(hero, team, -c(1:5))


###### Feature creation ####

### Individual hero winning percentage

hero_summary =
  hero_stats %>% 
  filter(team != 0) %>% 
  mutate(win = if_else(result == team, 1,0)) %>% 
  group_by(hero) %>% 
  summarise(win_perc = sum(win)/n()) 



### Co-occurence matrix for Team 1

heroes_coop_1 =
  hero_stats %>% 
  mutate(team = if_else(team == 1, 1,0)) %>% 
  select(1, 6,7) %>% 
  spread(hero, team) %>% 
  select(2:114) %>% 
  as.matrix(.) %>% 
  crossprod(.)  



### Co-occurence matrix for Team -1

heroes_coop_n1 =
  hero_stats %>% 
  mutate(team = if_else(team == -1, 1,0)) %>% 
  select(1, 6,7) %>% 
  spread(hero, team) %>% 
  select(2:114) %>% 
  as.matrix(.) %>% 
  crossprod(.)  



heroes_coop = heroes_coop_1 + heroes_coop_n1


### Winning Coop Matrix

heroes_win_matrix =
  hero_stats %>%
  mutate(win = if_else(result == team, 1,0)) %>% 
  select(1, 6,8) %>% 
  spread(hero, win) %>% 
  select(2:114) %>% 
  as.matrix(.)

heroes_coop_win = crossprod(heroes_win_matrix)

### Winning Percent matrix

heroes_coop_win_perc = heroes_coop_win/heroes_coop
diag(heroes_coop_win_perc) = 0

### convert to tibble

heroes_dyad_win = as_tibble(heroes_coop_win_perc, rownames = "hero")
colnames(heroes_dyad_win)[-1] = paste0("coop_", colnames(heroes_dyad_win)[-1])


#### Win prob against hero


#### playing against

heroes_against_1 =
  hero_stats %>%
  mutate(team = if_else(team == -1, 1, 0)) %>% 
  select(1, 6,7) %>% 
  spread(hero, team) %>% 
  select(2:114) %>% 
  as.matrix(.)  




heroes_against_n1 =
  hero_stats %>% 
  mutate(team = if_else(team == 1, 1, 0)) %>% 
  select(1, 6,7) %>% 
  spread(hero, team) %>% 
  select(2:114) %>% 
  as.matrix(.)  


heroes_against_matrix = crossprod(heroes_against_1, heroes_against_n1) + crossprod(heroes_against_n1, heroes_against_1)


#heroes_coop_loosing =
heroes_loss_matrix = hero_stats %>% 
  mutate(win = case_when(result == team ~ 0,
                         team != 0 & result != team ~ 1,
                         TRUE ~ 0) ) %>% 
  select(1, 6,8) %>% 
  spread(hero, win) %>% 
  select(2:114) %>% 
  as.matrix(.)

heroes_win_against = t(heroes_win_matrix) %*% heroes_loss_matrix

heroes_win_against_perc = heroes_win_against/heroes_against_matrix

### convert to tibble

heroes_vs_hero = as_tibble(heroes_win_against_perc, rownames = "hero")

colnames(heroes_vs_hero)[-1] = paste0("vs_", colnames(heroes_vs_hero)[-1])

### Function to recode Nan to NA
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

heroes_vs_hero[is.nan(heroes_vs_hero)] = 0


##### Combine data to hero feature data frame  of test data ####

hero_features = hero_summary %>% 
  inner_join(heroes_dyad_win) %>% 
  inner_join(heroes_vs_hero) 

hero_features[hero_features == 0] = NA


hero_stats = hero_stats %>% 
  filter(team != 0)

#### Create prediction data set for test data ####

dota_pred =
  hero_stats %>% 
  mutate(win = if_else(result == team, 1,0)) %>% 
  left_join(hero_features) %>% 
  group_by(ID, team) %>% 
  gather(co_vs, perc,-c(1:9)) %>% 
  filter(co_vs %in% paste0("coop_", hero) | co_vs %in% paste0("vs_", hero))




dota_pred =  
  dota_pred %>%
  na.omit(dota_pred) %>%
  group_by(ID, hero, cluster_id, game_mode, game_type) %>%
  separate(co_vs, "coop_vs", sep = "_") %>%
  summarise(
    team = first(team),
    result = first(result),
    hero_win = first(win_perc),
    coop = mean(perc[coop_vs == "coop"], na.rm = T),
    max_coop = max(perc[coop_vs == "coop"], na.rm = T),
    range_coop = max_coop - min(perc[coop_vs == "coop"], na.rm = T),
    vs = mean(perc[coop_vs == "vs"], na.rm = T),
    max_vs =  max(perc[coop_vs == "vs"], na.rm = T),
    range_vs = max_vs - min(perc[coop_vs == "vs"], na.rm = T)
  )


dota_pred = 
  dota_pred %>% 
  group_by(ID, team) %>% 
  arrange(hero_win) %>% 
  mutate(hero_no = paste0("hero_", team, "_",row_number())) %>%
  gather("metric", "value", -c(1:7, 15)) %>% 
  unite(hero_no, metric, col = "hero_feature", sep = "_") %>% 
  group_by(ID) %>% 
  select(-2, -team) %>% 
  spread(hero_feature, value)

save(knn_fit, xgb_fit, ranger_fit, hero_stats, hero_features, dota_pred, file = "dota2_data.RData")

#### predict results on Test data ####

dota_xgb_pred <- predict(xgb_fit, dota_pred)

dota_ranger_pred <- predict(ranger_fit, dota_pred)

dota_knn_pred <- predict(knn_fit, dota_pred)

#### Confusion matrices for quality evaluation ####

confu_matrix_knn = confusionMatrix(dota_knn_pred, dota_pred$result)

confu_matrix_ranger = confusionMatrix(dota_ranger_pred, dota_pred$result)

confu_xgb_knn = confusionMatrix(dota_xgb_pred, dota_pred$result)

