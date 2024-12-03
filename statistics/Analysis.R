#R version: 4.3.2
#packages used: lme4, ggplot2, Rmisc
#install.packages(c("lme4", "ggplot2", "Rmisc"))

##########################################
# Load data
##########################################

#file with individual moves per trial
moves <- read.csv("data/exp_processed/moves_w_alignment.csv")
#most important variables:
#condition: with or without AI
#generation: within a replication, from 0 to 3
#replication_idx: unique id for each replication
#solution_total_score: reward total per trial
#subject_id: unique per participant
#human_machine_match: judgment on move level, 1 if human and machine move match

#file with data on the player level
player <- read.csv("data/exp_processed/player.csv")

# file with coded player strategies
ratings_both <- read.csv("data/exp_strategies_coded/coded_strategies.csv")
#on (human) participant level, has the written strategies and our ratings for them
#loss_strategy codes 1 when strategy is present

##########################################
# Data preparation
##########################################

#aggregate moves to trial levels for performance
#points, only demonstration trials
demo <- subset(moves, moves$trial_type == "demonstration")
demo$branchID <- paste0(demo$replication_idx, demo$condition)
#this is the unique id for population

#different subsets for later
demo_agg <- subset(demo, demo$move_idx == 0)
demo_gen1plus <- subset(demo_agg, demo_agg$generation > 0)
demo_lastgen <- subset(demo_gen1plus, demo_gen1plus$generation == 4)
demo_firstgen <- subset(demo_gen1plus, demo_gen1plus$generation == 1)

# player ratings second strategy
player_ratings <- subset(ratings_both, ratings_both$written_strategy_idx == 1)

##########################################
# Descriptive Figures
##########################################

#some figures on performance
ci <- group.CI(solution_total_score ~ condition + generation, data = demo_agg)
ggplot(data = demo_agg, aes(x = generation, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()
#w/o AI
ci <- group.CI(solution_total_score ~ condition + generation, data = subset(demo_agg, demo_agg$ai_player == "False"))
ggplot(data = subset(demo_agg, demo_agg$ai_player == "False"), aes(x = generation, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()
runaverages <- aggregate(subset(demo_agg$solution_total_score, demo_agg$ai_player == "False"), by = list(subset(demo_agg$generation, demo_agg$ai_player == "False"), subset(demo_agg$replication_idx, demo_agg$ai_player == "False"), subset(demo_agg$condition, demo_agg$ai_player == "False")), FUN = mean)
ggplot(data = subset(demo_agg, demo_agg$ai_player == "False"), aes(x = generation, y = solution_total_score, color = condition)) + geom_point(aes(x = Group.1, y = x, color = Group.3), data = runaverages, size = 1, position= position_jitter(width = 0.07), alpha = 0.4) + stat_summary(geom = "point", fun = mean, size = 4, shape = 18) + coord_cartesian(ylim = c(1250,2500)) + xlab("Generation") + ylab("Average Reward") + scale_color_manual(values=c("orange", "blue"), labels=c("AI Tree", "Human Tree")) + theme_light() + theme(axis.text = element_text(size=12), axis.title = element_text(size = 14))
playeravg <- aggregate(subset(demo_agg$solution_total_score, demo_agg$ai_player == "False"), by = list(subset(demo_agg$generation, demo_agg$ai_player == "False"), subset(demo_agg$session_id, demo_agg$ai_player == "False"), subset(demo_agg$condition, demo_agg$ai_player == "False")), FUN = mean)
ggplot(data = subset(demo_agg, demo_agg$ai_player == "False"), aes(x = generation, y = solution_total_score, color = condition)) + geom_point(aes(x = Group.1, y = x, color = Group.3, group = Group.3), data = runaverages, size = 1.2, position= position_dodge(width = 0.4), alpha = 0.5) + stat_summary(aes(group = condition), geom = "point", fun = mean, size = 4, shape = 18, position = position_dodge(width = 0.4)) + coord_cartesian(ylim = c(1250,2500)) + xlab("Generation") + ylab("Average Reward") + scale_color_manual(values=c("orange", "blue"), labels=c("AI Tree", "Human Tree")) + theme_light() + theme(axis.text = element_text(size=12), axis.title = element_text(size = 14))
#gen0 excluding AI
ci <- group.CI(solution_total_score ~ condition + generation, data = subset(demo_agg, demo_agg$ai_player == "False"))
ggplot(data = subset(ci, ci$generation == 0), aes(x = condition, y = solution_total_score.mean, color = condition)) + geom_point() + geom_pointrange(aes(x = condition, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition)) + ylim(0,2700) + theme_light()
#max player score in gen0
ggplot(data = subset(player, player$generation == 0 & player$ai_player == "False"), aes(x = replication_idx, y = player_score, color = condition)) + stat_summary(geom = "point", fun = max) + ylim(0,2700) + theme_light()
mean(subset(player$player_score, player$generation == 0 & player$ai_player == "False" & player$condition == "w_ai"))
mean(subset(player$player_score, player$generation == 0 & player$ai_player == "False" & player$condition == "wo_ai"))
#individual improvement
individ <- subset(moves, moves$move_idx == 0)
individ_noAI <- subset(individ, individ$ai_player == "False")
individ_gen0 <- subset(individ_noAI, individ_noAI$generation == 0)
individ_nogen0 <- subset(individ, individ$generation > 0)
#plots
ci <- group.CI(solution_total_score ~ condition + trial_id, data = individ_gen0)
ggplot(data = individ_gen0, aes(x = trial_id, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = trial_id, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()
individ_nogen0 <- subset(individ_nogen0, individ_nogen0$trial_type %in% c("individual", "try_yourself", "demonstration"))
ci <- group.CI(solution_total_score ~ condition + trial_id, data = individ_nogen0)
ggplot(data = individ_nogen0, aes(x = trial_id, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = trial_id, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()
individ_noAI <- subset(individ_noAI, individ_noAI$trial_type %in% c("individual", "try_yourself", "demonstration"))
individ_noAI$gen <- ifelse(individ_noAI$generation > 0, "1-4", "0")
individ_noAI$trial_id <- ifelse(individ_noAI$trial_type == "demonstration" & individ_noAI$gen == "0", individ_noAI$trial_id + 10, individ_noAI$trial_id)
individ_noAI$trial_id <- ifelse(individ_noAI$trial_type == "individual" & individ_noAI$gen == "0" & individ_noAI$trial_id > 5, individ_noAI$trial_id * 1.5, individ_noAI$trial_id)
ggplot(data = individ_noAI, aes(x = trial_id, y = solution_total_score, color = condition, shape = gen)) + stat_summary(geom = "point", fun = mean) + theme_light()

##########################################
# Descriptive Analysis
##########################################

# Q: Is the loss strategy present in the written strategies before social learning?

#change in written strats
#t1 is always 0, t2 is 1 or 2
#this includes the t1 strategies
ratings1to4 <- subset(ratings_both, ratings_both$generation > 0)
ratings1to4$written_strategy_idx <- ifelse(ratings1to4$written_strategy_idx == 2, 1, ratings1to4$written_strategy_idx)
ci <- group.CI(loss_strategy ~ condition + written_strategy_idx, data = ratings1to4)
ggplot(data = ratings1to4, aes(x = written_strategy_idx, y = loss_strategy, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = written_strategy_idx, y = loss_strategy.mean, ymin = loss_strategy.lower, ymax = loss_strategy.upper, color = condition), data = ci) + theme_light()
unique(ave(ratings1to4$loss_strategy, ratings1to4$condition, ratings1to4$written_strategy_idx))

#include gen0
ratings_both$written_strategy_idx <- ifelse(ratings_both$written_strategy_idx == 2, 1, ratings_both$written_strategy_idx)
ratings_both$gen <- ifelse(ratings_both$generation > 0, "1-4", "0")
ratings_both$genchar <- as.character(ratings_both$generation)
ggplot(data = ratings_both, aes(x = written_strategy_idx, y = loss_strategy, color = condition, shape = gen)) + stat_summary(geom = "point", fun = mean) + theme_light()
ratings_gen0 <- subset(ratings_both, ratings_both$generation == 0)
unique(ave(ratings_gen0$loss_strategy, ratings_gen0$condition, ratings_gen0$written_strategy_idx))


##########################################
# Statistical Analysis
##########################################

#Statistical models on Performance
#hyp 1a
ci <- group.CI(solution_total_score ~ condition + generation, data = demo_gen1plus)
ggplot(data = demo_gen1plus, aes(x = generation, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()

demo_gen1plus$generation <- scale(demo_gen1plus$generation)
model <- lm(solution_total_score ~ condition + generation, data = demo_gen1plus)
summary(model)
model <- lmer(solution_total_score ~ condition + generation + (1|session_id), data = demo_gen1plus)
model <- lmer(solution_total_score ~ condition * generation + (1|session_id) + (generation|branchID), data = demo_gen1plus)
confint(model, method = "boot", verbose = TRUE)

#hyp1b
ci2 <- group.CI(solution_total_score ~ condition, data = demo_lastgen)
ggplot(data = demo_lastgen, aes(x = condition, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = condition, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci2) + ylim(0,3000) + theme_light()

model <- lm(solution_total_score ~ condition, data = demo_lastgen)
summary(model)
model <- lmer(solution_total_score ~ condition + (1|session_id), data = demo_lastgen)
model <- lmer(solution_total_score ~ condition + (1|session_id) + (1|branchID), data = demo_lastgen)
confint(model, method = "boot", verbose = TRUE)

#hyp2a
ci2 <- group.CI(solution_total_score ~ condition, data = demo_firstgen)
ggplot(data = demo_firstgen, aes(x = condition, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = condition, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci2) + ylim(0,3000) + theme_light()

model <- lm(solution_total_score ~ condition, data = demo_firstgen)
summary(model)
model <- lmer(solution_total_score ~ condition + (1|session_id), data = demo_firstgen)
model <- lmer(solution_total_score ~ condition + (1|session_id) + (1|branchID), data = demo_firstgen)
confint(model, method = "boot")
#singular, fit all optimizers
model.all <- allFit(model)
summary(model.all)

#stargazer(model, model2, model3, type = "text", column.labels = c("1a", "1b", "2a"), dep.var.caption =  "Prediction", dep.var.labels = "", digits = 1, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")
#stargazer(model, model2, model3, column.labels = c("1a", "1b", "2a"), dep.var.caption =  "Prediction", dep.var.labels = "", digits = 1, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")

#Statistical models on alignment
moves$human_machine_match <- ifelse(moves$human_machine_match == "True", 1, 0)
ci <- group.CI(human_machine_match ~ condition + generation, data = moves)
ggplot(data = moves, aes(x = generation, y = human_machine_match, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = human_machine_match.mean, ymin = human_machine_match.lower, ymax = human_machine_match.upper, color = condition), data = ci) + ylim(0,1) + theme_light()
#w/o AI
ci <- group.CI(human_machine_match ~ condition + generation, data = subset(moves, moves$ai_player == "False"))
ggplot(data = subset(moves, moves$ai_player == "False"), aes(x = generation, y = human_machine_match, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = human_machine_match.mean, ymin = human_machine_match.lower, ymax = human_machine_match.upper, color = condition), data = ci) + ylim(0,1) + theme_light()

aldemo <- subset(moves, moves$trial_type == "demonstration")
aldemo$branchID <- paste0(aldemo$replication_idx, aldemo$condition)
aldemo_gen1plus <- subset(aldemo, aldemo$generation > 0)
aldemo_lastgen <- subset(aldemo_gen1plus, aldemo_gen1plus$generation == 4)
aldemo_firstgen <- subset(aldemo_gen1plus, aldemo_gen1plus$generation == 1)

#hyp 1a
aldemo_gen1plus$generation <- scale(aldemo_gen1plus$generation)
model <- glm(human_machine_match ~ condition + generation, data = aldemo_gen1plus, family = binomial(link = "logit"))
summary(model)
model <- glmer(human_machine_match ~ condition * generation + (1|session_id) + (generation|branchID), data = aldemo_gen1plus, family = binomial(link = "logit"))
#warning: this is a very large model and 1000 bootstraps will take several hours!
confint(model, method = "boot", verbose = TRUE)

#hyp1b
model <- glm(human_machine_match ~ condition, data = aldemo_lastgen, family = binomial(link = "logit"))
summary(model)
model <- glmer(human_machine_match ~ condition + (1|session_id) + (1|branchID), data = aldemo_lastgen, family = binomial(link = "logit"))
confint(model, method = "boot", verbose = TRUE)

#hyp2a
model <- glm(human_machine_match ~ condition, data = aldemo_firstgen, family = binomial(link = "logit"))
summary(model)
model <- glmer(human_machine_match ~ condition + (1|session_id) + (1|branchID), data = aldemo_firstgen, family = binomial(link = "logit"))
confint(model, method = "boot", verbose = TRUE)

model.all <- allFit(model)
summary(model.all)

#stargazer(model, model2, model3, type = "text", column.labels = c("1a", "1b", "2a"), dep.var.caption =  "Prediction", dep.var.labels = "", digits = 3, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")
#stargazer(model, model2, model3, column.labels = c("1a", "1b", "2a"), dep.var.caption =  "Prediction", dep.var.labels = "", digits = 3, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")

#written strategies

#hyp2b
player_ratings$branchID <- paste0(player_ratings$replication_idx, player_ratings$condition)
ratings_gen1plus <- subset(player_ratings, player_ratings$generation > 0)

ci <- group.CI(loss_strategy ~ condition + generation, data = player_ratings)
ggplot(data = player_ratings, aes(x = generation, y = loss_strategy, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = loss_strategy.mean, ymin = loss_strategy.lower, ymax = loss_strategy.upper, color = condition), data = ci) + ylim(0,1) + theme_light()
runaverages <- aggregate(player_ratings, by = list(player_ratings$generation, player_ratings$replication_idx, player_ratings$condition), FUN = mean)
ggplot(data = player_ratings, aes(x = generation, y = loss_strategy, color = condition)) + geom_point(aes(x = Group.1, y = loss_strategy, color = Group.3, group = Group.3), data = runaverages, size = 1.2, position= position_dodge(width = 0.4), alpha = 0.5) + stat_summary(aes(group = condition), geom = "point", fun = mean, size = 4, shape = 18, position = position_dodge(width = 0.4)) + coord_cartesian(ylim = c(0,1)) + xlab("Generation") + ylab("Loss Strategy") + scale_color_manual(values=c("orange", "blue"), labels=c("AI Tree", "Human Tree")) + theme_light() + theme(axis.text = element_text(size=12), axis.title = element_text(size = 14))

ratings_gen1plus$generation <- scale(ratings_gen1plus$generation)
model <- glm(loss_strategy ~ condition + generation, data = ratings_gen1plus, family = binomial(link = "logit"))
summary(model)
model <- glmer(loss_strategy ~ condition * generation + (generation|branchID), data = ratings_gen1plus, family = binomial(link = "logit"))
confint(model, method = "boot", verbose = TRUE)
model.all <- allFit(model)
summary(model.all)

#stargazer(model, type = "text", column.labels = "2b", dep.var.caption =  "Prediction", dep.var.labels = "", digits = 3, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")
#stargazer(model, column.labels = "2b", dep.var.caption =  "Prediction", dep.var.labels = "", digits = 3, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")
