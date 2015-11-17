setwd("~/Dropbox/ABACUS/Workspace/LeapArticulatorQt")
all <- read.csv("all_scores_bics_nstates_by_phase.csv")
all$id <- factor(all$id)
all$phase <- factor(all$phase)
all$condition <- factor(all$condition)
all$phase_order <- factor(all$phase_order)

library(lme4)
library(lmerTest)
library(MCMCglmm)

r2.corr.mer <- function(m) {
  lmfit <-  lm(model.response(model.frame(m)) ~ fitted(m))
  summary(lmfit)$r.squared
}


# model.null <- lmer(score ~ nstates_amp_and_freq +  
#                      (1 | id) + (1 |condition:phase), data=all, REML=F)
model1 <- lmer(score ~ nstates_amp_and_freq_n:phase_order:phase +
                 (1 | id) , data=all, REML=F)
model2 <- lmer(score ~ nstates_amp_and_freq_n:phase +
                 (1 | id) , data=all, REML=F)
anova(model1,model2)

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
# library(lattice)
# dotplot(ranef(model1, condVar=T))
# coef(model1)
# 
# cond.phase2phase <- function(txt) {
#   if (txt == "1:0") {
#     return("1:1")
#   }
#   else if(txt == "1:1") {
#     return("1:2")
#   }
#   else if(txt == "1:2") {
#     return("2:2")
#   } 
#   else if (txt == "2:0") {
#     return("1:1")
#   }
#   else if(txt == "2:1") {
#     return("2:2")
#   }
#   else if(txt == "2:2") {
#     return("1:2")
#   }
# }
# 
# rand_effects <- ranef(model1, condVar=T)
# qq <- attr(rand_effects[[2]], "postVar")
# rand.intercepts <- rand_effects$`condition:phase`
# # rand.intercepts <- rand.intercepts
# phases <- simplify2array(lapply(rownames(rand.intercepts), cond.phase2phase))
# intercepts <- c(mean(rand.intercepts[c("1:0","2:0"),]), rand.intercepts[c("1:1","1:2","2:1","2:2"),])
# labels <- c(expression(beta[1]), "$\\beta_4", "$\\beta_3","$\\beta_2","$\\beta_5")
# df<-data.frame(Intercepts=intercepts,
#                sd.interc=sqrt(qq[,,2:length(qq)]),
#                lev.names=factor(phases[c(1,2,3,5,6)]),
#                Order=factor(c(1,2,3,2,3)))
# 
# cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
# 
# p <- ggplot(data=df, 
#             aes(x=lev.names, y=Intercepts, colour=Order)
#             ) + geom_point()
# p <- p + geom_text(aes(label=paste("beta[", c(1,4,3,2,5), "]", sep=""), 
#                        hjust=1.5), 
#                    parse = TRUE)
# level_names <- c("1:1",
#                  "1:2",
#                  "2:2")
# p <- p + scale_x_discrete("Phases", labels=level_names) + ylab("")
# p <- p + geom_errorbar(data=df, 
#                        aes(ymin=Intercepts-sd.interc, 
#                            ymax=Intercepts+sd.interc, 
#                            width=.3)) 
# p + scale_colour_manual(values=cbPalette)# + coord_flip()

library(coefplot)
library(stringr)
# newNames = c("1st:mismatchF","2nd:mismatchF","3rd:mismatchF", 
#              "2nd:mismatchT", "3rd:mismatchT")
oldNames = c("beta_1","beta_2","beta_3","beta_4", "beta_5")
newNames = simplify2array(lapply(str_replace_all(oldNames, "_([:digit:])", "[\\1]"), function(x) {parse(text=x)}))
# oldNames = list("$\\alpha_0$","$\\beta_1$ (1:1)","$\\beta_2$ (1:2)","$\\beta_3$ (2:2)")
# newNames = list("$\\beta_1$ (1:1,1st)","$\\beta_4$ (1:2,2nd)","$\\beta_5$ (1:2,3rd)","$\\beta_2$ (2:2,2nd)","$\\beta_3$ (2:2,3rd)")
# newNames = c(parse(text="beta[1]"),parse(text="beta[4]"),parse(text="beta[5]"),parse(text="beta[2]"),parse(text="beta[3]"))
# newNames = c(parse(text="beta[1]"),parse(text="beta[4]"),parse(text="beta[5]"),parse(text="beta[2]"),parse(text="beta[3]"))
ordering = factor(c("1st","2nd","3rd","2nd","3rd"))
phase = factor(c("1:1", "1:2", "1:2", "2:2", "2:2"))
effect_sd = summary(model1)$coefficients[c(8,9,10,11,12)]
effect = fixef(model1)[c(2,3,4,5,6)]
ci.low = effect-effect_sd
ci.hi = effect+effect_sd
# oldNames <- factor(x=oldNames, levels=oldNames[order(oldNames)], labels=newNames[order(oldNames)])

# newNames = apply(new)
#names(newNames) = names(coef(model1)$id)[2:6]
df <- data.frame(coefs=effect, ci.low=ci.low, ci.hi = ci.hi, ordering=ordering, 
                 label=oldNames, phase=phase)
# df$label <- factor(df$label, levels=df$label[c(1,4,5,2,3)])
# df <- df[with(df, order(label)),]
# df$label <- factor(df$label, as.character(df$label))

#p <- coefplot(model1, ylab="Coefficients", intercept=FALSE, parse=T)
p <- ggplot(aes(y=coefs, x=label, color=ordering, shape=phase), data=df) + geom_point(size=5) + ggtitle("Fixed Effects")
p <- p + geom_errorbar(aes(ymin=ci.low, ymax=ci.hi, height=.33), width=.33, size=1, data=df)
p <- p + theme(axis.text.x = element_text(size=14),axis.text.y = element_text(size=14))
p <- p + scale_x_discrete(labels=newNames)# + ylim(newNames)
p + labs(x= "Value", y="Coefficient") 


# the Bayesian MCMC stuff from here
library(MCMCglmm)
model.glmm <- MCMCglmm(score ~ nstates_amp_and_freq_n:phase_order:phase, random=~id, data=all, nitt=1000000, thin=500,
                       pr = TRUE)

# we don't take the random effects of id, the intercept or the main predictor
samples <- model.glmm$Sol[,c(25:30)]
means <- apply(samples, 2, mean)
medians <- apply(samples, 2, median)
CI.low <- list()
CI.high <- list()
# calculate CIs
for (i in 1:dim(samples)[[2]]) {
  CI.low[[i]] <- quantile(samples[,i], c(.025))[[1]];
  CI.high[[i]] <- quantile(samples[,i], c(.975))[[1]];
}

df <- data.frame(mean=means, median=medians, CI.low=as.numeric(CI.low), CI.high=as.numeric(CI.high), name=names(means))

p <- ggplot(data=df, 
            aes(x=name, y=median)
) + geom_errorbar(aes(ymin=CI.low, ymax=CI.high)) 

p <- p + geom_point()# + geom_point(data=df, aes(x=name, y=CI.low, color="red"))+ geom_point(data=df, aes(x=name, y=CI.high, color="green"))

# p <- p + ggplot(data=df, aes(x=name, y=CI.low, color="red"))
# p
# p
p + geom_point()

library(tikzDevice)
newNames = simplify2array(str_replace_all(oldNames, "(.+)", "$\\\\\\1$"))
image_filename <- '/home/kerem/Dropbox/ABACUS/Publications/journalpapes/images/fixed_effects.tex'
tikz(image_filename, standAlone = TRUE, width = 3.7, height = 3)
p <- ggplot(aes(y=coefs, x=label, color=ordering, shape=phase), data=df) + geom_point(size=5) + ggtitle("Fixed Effects")
p <- p + geom_errorbar(aes(ymin=ci.low, ymax=ci.hi, height=.33), width=.33, size=1, data=df)
p <- p + theme(axis.text.x = element_text(size=14),axis.text.y = element_text(size=14))
p <- p + scale_x_discrete(labels=newNames)# + ylim(newNames)
p + labs(y= "Value", x="Coefficient") 
# newNames = list(parse(text="beta[1]") + "(1:1,1st)","$\\beta_4$ (1:2,2nd)","$\\beta_5$ (1:2,3rd)","$\\beta_2$ (2:2,2nd)","$\\beta_3$ (2:2,3rd)")
# names(newNames) = names(coef(model1)$id)[2:6]
# coefplot(model1, newNames=newNames, ylab="Coefficients", intercept=FALSE)
dev.off()
library(tools)
texi2pdf(file=image_filename, clean=TRUE)
outfile <- paste(paste(str_split(image_filename, "/")[[1]][1:8],collapse="/"), "fixed_effects.pdf", sep="/")
file.copy(from="fixed_effects.pdf", to=outfile, overwrite = TRUE)
# file.remove("intercepts.pdf")
