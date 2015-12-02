  setwd("~/Dropbox/ABACUS/Workspace/LeapArticulatorQt/logs/logs/discrete")
  all <- read.csv("all_scores_bics_nstates_by_phase.csv")
  all$id <- factor(all$id)
  all$condition <- factor(all$condition)
  all$phase[all$phase == 0] <- "1to1"
  all$phase[all$phase == 1 & all$condition == 1] <- "1to2"
  all$phase[all$phase == 2 & all$condition == 1] <- "2to2"
  
  all$phase[all$phase == 1 & all$condition == 2] <- "2to2"
  all$phase[all$phase == 2 & all$condition == 2] <- "1to2"
  all$phase <- factor(all$phase)
  
  all$phase_order[all$phase == "1to1"] <- "1st"
  all$phase_order[all$phase == "1to2" & all$condition == 1] <- "2nd"
  all$phase_order[all$phase == "2to2" & all$condition == 1] <- "3rd"
  all$phase_order[all$phase == "2to2" & all$condition == 2] <- "2nd"
  all$phase_order[all$phase == "1to2" & all$condition == 2] <- "3rd"
  all$phase_order = factor(all$phase_order)
  
  all$iconic[all$phase == "1to1" | all$phase == "2to2"] = T
  all$iconic[all$phase == "1to2"] = F
  
  library(lme4)
  library(lmerTest)
  require(lattice)
  # library(MCMCglmm)
  
  r2.corr.mer <- function(m) {
    lmfit <-  lm(model.response(model.frame(m)) ~ fitted(m))
    summary(lmfit)$r.squared
  }
  
  
  # model.null <- lmer(score ~ 
  #                      (1 | id), data=all, REML=F)
  # model.null <- lmer(score ~ nstates_amp_and_mel_n : phase +  
  #                  (1 | id) , data=all, REML=F)
  model.null <- lmer(score ~ nstates_amp_and_mel_n:phase  + 
                       #                    - nstates_amp_and_mel_n - phase_order + 
                       (1 | id ) , data=all, REML=F) 
  model1 <- lmer(score ~ nstates_amp_and_mel_n:phase_order:phase  +  
                   #                    - nstates_amp_and_mel_n - phase_order + 
                   (1 | id), data=all, REML=F)
  model2 <- lmer(score ~ nstates_amp_and_mel_n:phase_order:phase + 
                   #                    - nstates_amp_and_mel_n - phase_order + 
                   (1 | id), data=all, REML=F)
  
  random_effects <- ranef(model1, postVar=T)
  intercepts_by_id <- random_effects[[1]][[1]]
  slopes_by_id <- random_effects[[1]][[2]]
  intercepts_by_phase <- random_effects[[2]][[1]]
  slopes_by_phase <- random_effects[[2]][[2]]
  n_of_ids <- length(unique(all$id))
  total_scores <- all$total_score[1:n_of_ids * 3][1:n_of_ids]
  phase111_scores <- all[all$phase == "1to1" & all$phase_order == "1st", 'score']
  phase212_scores <- all[all$phase == "1to2" & all$phase_order == "2nd", 'score']
  phase312_scores <- all[all$phase == "1to2" & all$phase_order == "3rd", 'score']
  phase222_scores <- all[all$phase == "2to2" & all$phase_order == "2nd", 'score']
  phase322_scores <- all[all$phase == "2to2" & all$phase_order == "3rd", 'score']
  random_slopes <- coef(model1)[[1]]$nstates_amp_and_mel_n
  
  
  # summary(model.null)
  summary(model1)
  # summary(model2)
  # hist(residuals(model2))
  # summary(model1)
  r2.corr.mer(model.null)
  r2.corr.mer(model1)
  # r2.corr.mer(model2)
  anova(model.null, model1)
  # anova(model1, model2)
  dotplot(ranef(model.null, condVar=T))
  # coef(model1)
  # plot(c(0,1), c(0, 1)); mapply(a=slopes_by_id, FUN=abline, b=intercepts_by_id)
  model1 <- lmer(score ~ nstates_amp_and_mel_n:phase_order:phase + 
                   (1 | id), data=all, REML=F)
  model2 <- lmer(score ~ nstates_amp_and_mel_n + (1|condition:phase) + (1|id), data=all, REML=F)
  anova(model1, model2)
  summary(model1)
  
  library(coefplot)
  newNames = c("1st:mismatchF","2nd:mismatchF","3rd:mismatchF", 
               "2nd:mismatchT", "3rd:mismatchT")
  oldNames = c("beta_1","beta_2","beta_3","beta_4", "beta_5")
  # oldNames = list("$\\alpha_0$","$\\beta_1$ (1:1)","$\\beta_2$ (1:2)","$\\beta_3$ (2:2)")
  newNames = list("$\\beta_1$ (1:1,1st)","$\\beta_2$ (1:2,2nd)","$\\beta_3$ (1:2,3rd)","$\\beta_4$ (2:2,2nd)","$\\beta_5$ (2:2,3rd)")
  newNames = c(parse(text="beta[1]"),parse(text="beta[2]"),parse(text="beta[3]"),parse(text="beta[4]"),parse(text="beta[5]"))
  order = factor(c("1st","2nd","3rd","2nd","3rd"))
  phase = factor(c("1:1", "1:2", "1:2", "2:2", "2:2"))
  effect_sd = summary(model1)$coefficients[8:12]
  effect = fixef(model1)[2:6]
  ci.low = effect-effect_sd
  ci.hi = effect+effect_sd
  
  # newNames = apply(new)
  #names(newNames) = names(coef(model1)$id)[2:6]
  df <- data.frame(coefs=effect, ci.low=ci.low, ci.hi = ci.hi, order=order, label=oldNames, phase=phase)
  
  #p <- coefplot(model1, ylab="Coefficients", intercept=FALSE, parse=T)
  p <- ggplot(aes(x=coefs, y=label, color=order, shape=phase),title="Fixed Effects", data=df) + geom_point(size=5)
  p <- p + geom_errorbarh(aes(xmin=ci.low, xmax=ci.hi, height=.33), size=1, data=df)
  p <- p + scale_y_discrete(name="Coefficients", labels=newNames) + theme(axis.text.y = element_text(size=14),axis.text.x = element_text(size=14))
  p + coord_flip() + scale_colour_grey() + scale_fill_grey() + theme(
    panel.background = element_rect(fill = "white",
                                    colour = "white",
                                    size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                    colour = "gray"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                    colour = "gray"))
  
  library(coefplot)
  newNames = c("Baseline","1st:mismatchF","2nd:mismatchF","3rd:mismatchF", 
               "2nd:mismatchT", "3rd:mismatchT")
  newNames = list("$\\alpha_0$","$\\beta_1$ (1:1)","$\\beta_2$ (1:2)","$\\beta_3$ (2:2)")
  names(newNames) = names(fixef(model1))
  coefplot(model1, newNames=newNames, coefficients=names(newNames)[2:4], ylab="Coefficients")
  
  library(tikzDevice)
  newNames = simplify2array(str_replace_all(oldNames, "(.+)", "$\\\\\\1$"))
  image_filename <- '/home/kerem/Dropbox/ABACUS/Publications/journalpapes/images/fixed_effects_discrete.tex'
  tikz(image_filename, standAlone = TRUE, width = 3.7, height = 3)
  p <- ggplot(aes(y=coefs, x=label, color=ordering, shape=phase), data=df) + geom_point(size=5) + ggtitle("Fixed Effects")
  p <- p + geom_errorbar(aes(ymin=ci.low, ymax=ci.hi, height=.33), width=.33, size=1, data=df)
  p <- p + theme(axis.text.x = element_text(size=14),axis.text.y = element_text(size=14))
  p <- p + scale_x_discrete(labels=newNames)# + ylim(newNames)
  p + labs(y= "Value", x="Coefficient") +  scale_colour_grey() + scale_fill_grey() + theme(
    panel.background = element_rect(fill = "white",
                                    colour = "white",
                                    size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                    colour = "gray"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                    colour = "gray"))
  # newNames = list(parse(text="beta[1]") + "(1:1,1st)","$\\beta_4$ (1:2,2nd)","$\\beta_5$ (1:2,3rd)","$\\beta_2$ (2:2,2nd)","$\\beta_3$ (2:2,3rd)")
  # names(newNames) = names(coef(model1)$id)[2:6]
  # coefplot(model1, newNames=newNames, ylab="Coefficients", intercept=FALSE)
  dev.off()
  library(tools)
  texi2pdf(file=image_filename, clean=TRUE)
  outfile <- paste(paste(str_split(image_filename, "/")[[1]][1:8],collapse="/"), "fixed_effects_discrete.pdf", sep="/")
  file.copy(from="fixed_effects_discrete.pdf", to=outfile, overwrite = TRUE)
  file.remove("fixed_effects_discrete.pdf")
  