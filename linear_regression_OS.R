setwd("~/Dropbox/ABACUS/Workspace/LeapArticulatorQt/logs/logs/orange_squares")
all <- read.csv("all_scores_bics_nstates_by_phase.csv")
all$id <- factor(all$id)
all$iconic <- all$phase
all$phase <- factor(all$phase)
all$phase_order <- factor(all$phase_order)
all$iconic[all$iconic == 0] <- -1
all$iconic[all$iconic == 1] <- 0
all$iconic[all$iconic == 2] <- 0
all$iconic[all$iconic == -1] <- 1
all$n_meanings[all$phase == 0] <- 6
all$n_meanings[all$phase == 1] <- 36
all$n_meanings[all$phase == 2] <- 216
all$n_meanings_n <- normalize(all$n_meanings)
# all$log_nstates_amp_and_mel <- log(all$nstates_amp_and_mel)
# all$log_nstates_amp_and_freq <- log(all$nstates_amp_and_freq)
# all$log_nstates_xy <- log(all$nstates_xy)
# all$log_nstates_amp_and_mel_n <- log(all$nstates_amp_and_mel_n)
# all$log_nstates_amp_and_freq_n <- log(all$nstates_amp_and_freq_n)
# all$log_nstates_xy_n <- log(all$nstates_xy_n)
# all$phase <- factor(all$phase)

library(lme4)
library(lmerTest)
library(MCMCglmm)

r2.corr.mer <- function(m) {
  lmfit <-  lm(model.response(model.frame(m)) ~ fitted(m))
  summary(lmfit)$r.squared
}


model.null <- lmer(score ~ nstates_amp_and_mel + phase - 1 + 
                     (1 | id) , data=all, REML=F)
model1 <- lmer(score ~ nstates_amp_and_mel + phase 
                 + (1|id), 
               data=all, REML=F)

# model.glmm <- MCMCglmm(score ~ nstates_amp_and_mel:phase_order:phase - 1, random=~id, data=all, nitt=500000, thin=500,
#                        pr = TRUE)
# summary(model.glmm)

summary(model.null)
summary(model1)
# summary(model2)
# hist(residuals(model2))
# summary(model1)
# r2.corr.mer(model.null)
r2.corr.mer(model.null)
r2.corr.mer(model1)
# r2.corr.mer(model2)
anova(model.null, model1)
# anova(model1, model2)
# dotplot(ranef(model1, postVar=T))
# coef(model1)