setwd("~/Dropbox/ABACUS/Workspace/LeapArticulator")
all <- read.csv("all_scores_bics_nstates_by_phase.csv")
all$id <- factor(all$id)

library(lme4)
library(lmerTest)
library(MCMCglmm)

r2.corr.mer <- function(m) {
  lmfit <-  lm(model.response(model.frame(m)) ~ fitted(m))
  summary(lmfit)$r.squared
}


# model.null <- lmer(score ~ nstates_amp_and_freq +  
#                      (1 | id) + (1 |condition:phase), data=all, REML=F)
model1 <- lmer(score ~ nstates_amp_and_freq_n +
                     (1 | id) + (1 |condition:phase) , data=all, REML=F)


# summary(model.null)
summary(model1)
# summary(model2)
# hist(residuals(model2))
# summary(model1)
# r2.corr.mer(model.null)
r2.corr.mer(model1)
# r2.corr.mer(model2)
# anova(model.null, model1)
# anova(model1, model2)
dotplot(ranef(model1, postVar=T))
coef(model1)