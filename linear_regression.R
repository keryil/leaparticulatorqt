setwd("~/Dropbox/ABACUS/Workspace/LeapArticulatorQt/logs/logs/first_exp_data")
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
r2.corr.mer(model1)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
    # grid.text("I AM THE TITLE", viewport(layout.pos.row=1, layout.pos.col=1:2))
  }
}

draw_model_coefs<- function(model, subset) {
  library(coefplot)
  library(stringr)
  # subsets = c(...)
  # draw_one <- function(model, subset) {
    phase_template = c("1:1", "1:2", "1:2", "2:2", "2:2")
    name_template = c("beta_1","beta_2","beta_3","beta_4", "beta_5")
    order_template = c("1st","2nd","3rd","2nd","3rd")
    oldNames = name_template[subset]
    newNames = factor(phase_template[subset], levels=phase_template[subset])
    # newNames = simplify2array(lapply(str_replace_all(oldNames, "_([:digit:])", "[\\1]"), function(x) {parse(text=x)}))
    ordering = factor(order_template[subset], levels=order_template[subset])
    phase = factor(phase_template[subset], levels=phase_template[subset])
    effect_sd = summary(model)$coefficients[c(8,9,10,11,12)][subset]
    effect = as.numeric(fixef(model)[c(2,3,4,5,6)])[subset]
    print(effect)
    ci.low = effect-effect_sd
    ci.hi = effect+effect_sd
    
    df <- data.frame(coefs=effect, ci.low=ci.low, ci.hi = ci.hi, ordering=ordering, 
                     label=oldNames, phase=phase)
    print(c("Data frame: ", df))
    
    p <- ggplot(aes(y=coefs, x=phase), data=df) + geom_point(size=5) #+ ggtitle("Fixed Effects")
    p <- p + geom_errorbar(aes(ymin=ci.low, ymax=ci.hi, height=.33), width=.33, size=1, data=df)
    p <- p + theme(axis.text.x = element_text(size=14),axis.text.y = element_text(size=14))
    p <- p + scale_x_discrete(labels=newNames)# + ylim(newNames)
    p <- p + labs(x= "Phase", y="Coefficient") +  scale_colour_grey() + scale_fill_grey() + theme(
      panel.background = element_rect(fill = "white",
                                      colour = "white",
                                      size = 0.5, linetype = "solid"),
      panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                      colour = "gray"), 
      panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                      colour = "gray")) 
    p <- p + ylim(c(-.047,.3)) #+ theme(axis.title.x = element_blank()) 
    return(p)
  # }
#   plots <- array(dim=length(subsets))
#   print("Empty plots array:")
#   print(plots)
#   for (i in 1:length(subsets)) {
#     subset <- subsets[[i]]
#     plt <- draw_one(model, subset)
#     print("Inserting:")
#     print(plt)
#     plots[[i]] <- p
#   }
#   return(plots)
}
draw_model_coefs_separately <- function(model, subset1, subset2) {
  p1 <- draw_model_coefs(model, subset=subset1) + ggtitle("Ordering 1")
  p2 <- draw_model_coefs(model, subset=subset2) + ggtitle("Ordering 2")
  return(multiplot(p1, p2, cols=2))
}
draw_model_coefs_separately(model1, c(1,2,5), c(1,4,3))
# draw_model_coefs(model1)

# grid.newpage()
# grid.draw(rbind(ggplotGrob(p1), ggplotGrob(p2), size = "last"))

library(tikzDevice)
# newNames = simplify2array(str_replace_all(oldNames, "(.+)", "$\\\\\\1$"))
image_filename <- '/home/kerem/Dropbox/ABACUS/Publications/journalpapes/images/fixed_effects_all_separate.tex'
tikz(image_filename, standAlone = TRUE, width = 3.7, height = 3)
draw_model_coefs_separately(model1, c(1,2,5), c(1,4,3))
# p <- ggplot(aes(y=coefs, x=label, color=ordering, shape=phase), data=df) + geom_point(size=5) + ggtitle("Fixed Effects")
# p <- p + geom_errorbar(aes(ymin=ci.low, ymax=ci.hi, height=.33), width=.33, size=1, data=df)
# p <- p + theme(axis.text.x = element_text(size=14),axis.text.y = element_text(size=14))
# p <- p + scale_x_discrete(labels=newNames)# + ylim(newNames)
# p + labs(y= "Value", x="Coefficient")  +  scale_colour_grey() + scale_fill_grey() + theme(
#   panel.background = element_rect(fill = "white",
#                                   colour = "white",
#                                   size = 0.5, linetype = "solid"),
#   panel.grid.major = element_line(size = 0.5, linetype = 'solid',
#                                   colour = "gray"), 
#   panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
#                                   colour = "gray"))
# newNames = list(parse(text="beta[1]") + "(1:1,1st)","$\\beta_4$ (1:2,2nd)","$\\beta_5$ (1:2,3rd)","$\\beta_2$ (2:2,2nd)","$\\beta_3$ (2:2,3rd)")
# names(newNames) = names(coef(model1)$id)[2:6]
# coefplot(model1, newNames=newNames, ylab="Coefficients", intercept=FALSE)
dev.off()
library(tools)
texi2pdf(file=image_filename, clean=TRUE)
outfile <- paste(paste(str_split(image_filename, "/")[[1]][1:8],collapse="/"), "fixed_effects_all_separate.pdf", sep="/")
file.copy(from="fixed_effects_all_separate.pdf", to=outfile, overwrite = TRUE)
# file.remove("intercepts.pdf")
