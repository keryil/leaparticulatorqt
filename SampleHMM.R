# TODO: Okay, now this observations as lists-of-lists is not working too well. I need to find a way to 
# delineate every signal, but I also need a way of presenting per-person signal repertoires for training. 
# This will probably require multiple calls to the HMMfit() function.
require("depmixS4")
require("data.table")

fitHMMtoPhaseDepmix <- function(file_id, nStates, iter=100, dis="NORMAL", phase=NULL, nMixt=-1, take_vars=c('x','y')) {
  
  source("~/Dropbox/ABACUS/Workspace/LeapArticulator/Data Analysis.R")
  
  list[responses, tests, images] <- readLogFile(file_id)
  #   setwd("~/Dropbox/ABACUS/Workspace/LeapArticulator/")
  # print(images)
  d <- list()
  firsts = responses[responses$phase == phase & responses$data_index == 0,]
  #   View(firsts)
  counter = 1
  for(image in unique(firsts$image)) {
    for(practice in 0:1) {
      #         image = images[image_id,]
      #         print(image)
#       cat("image:", image, "phase:", phase, "practice:", practice, "\n")
      indices = which(responses$image == image & responses$is_practice == practice & responses$phase == phase & responses$data_index == 0)
      #         cat("Indices:", indices)
      first = responses[indices,'X']
      if (length(first) == 0) {
        print(paste("No responses for", image, "at phase", phase, "(practice?", practice, ")"))
        next
      }
      nxt = firsts[firsts$X > first,]
      print(nxt[1,"X"])
#       cat("Length of next:", nrow(nxt), "\n")
      #       exit()
      if(nrow(nxt) < 1) {
#         print("Zero length..................")
        nxt = tail(responses)
      }
#       cat("First index:", first, "\n")
#       cat("Last index:", nxt[1,"X"], "\n")
      #         print(paste("Interval: ", print(first:(nxt-1))))
      traj = responses[responses$X %in% first:(nxt[1,"X"]-1),]
      selected <- data.matrix(traj)
      #         View(traj)
      d[[counter]] <- selected
      cat("Trajectory",counter, "processed.", "\n")
      counter = counter + 1
    }
  }
#   print(length(d[1]))
  #   print(summary(d))
  # fit HMM
  #  d = d[-(which(sapply(d,is.null),arr.ind=TRUE))]
  if (nMixt != -1) {
    # print("Mixture model")
    hmm <- HMMFit(d, nStates=nStates, dis=dis, asymptCov=FALSE, nMixt=nMixt, control=list(verbose=0, iter=iter))
    path = list()
    for(trajectory in d){
      path = c(path, viterbi(hmm, trajectory))
    }
  }
  else {
    # print("Normal model")
    # print(summary(d))
    hmm <- HMMFit(d, nStates=nStates, dis=dis, asymptCov=FALSE, control=list(verbose=0, iter=iter))
    #print("Trained!")
    #     path = list()
    #     for(trajectory in d){
    #       path = c(path, viterbi(hmm, trajectory))
    #     }
    #     variances = hmm$HMM$distribution$cov
    #     transmat = hmm$HMM$transMat
    #     means = hmm$HMM$distribution$mean
  }
  
  return(list(hmm, d))
}

probabilityRow <- function(size) {
  row = runif(size)
  return(row/sum(row))
}

depmixUnivariateHMM <- function(responses, nStates, phase=1, take_vars='x') {
  list[d,lengths] <- prepareDataDepmix(responses,phase=phase,take_vars=take_vars)
  rModels <- list()
  transition <- list()
#   data <- data.frame(data)
#   rm(y)
  for(state in 1:nStates){  
    print(state)
    rModels[[state]] <- list(GLMresponse(x~1,family=gaussian()))
    transition[[state]] <- transInit(~1,nstates=nStates,pstart=probabilityRow(nStates),data=d)
  }
  
  inMod <- transInit(~1,ns=nStates,pstart=probabilityRow(nStates),data=d)
  mod <- makeDepmix(response=rModels,transition=transition,prior=inMod)
  return(mod)
}

depmixBivariateHMM <- function(responses, nStates, phase=1, take_vars=c('x','y')) {
  list[data,lengths] <- prepareDataDepmix(responses,phase=phase,take_vars=take_vars)
  rModels <- list()
  transition <- list()
  #   data <- data.frame(data)
  #   rm(y)
#   rModels[[1]] <- list(MVNresponse(y~1))
#   rModels[[2]] <- list(MVNresponse(y~1))
  for(state in 1:nStates){ 
    rModels[[state]] <- list(MVNresponse(data$x~1),MVNresponse(data$y~1)) 
    print(state)
    transition[[state]] <- transInit(~1,nstates=nStates,pstart=probabilityRow(nStates),data=data)
    density <- dens(transition[[state]])
    print(str(density))
    print(head(density))
  }
  
  inMod <- transInit(~1,ns=nStates,pstart=probabilityRow(nStates),data=data)
  mod <- makeDepmix(response=rModels,transition=transition,prior=inMod)
  return(mod)
}

prepareDataDepmix <- function(responses, phase, take_vars=c("x","y")) {
  d <- NULL
  firsts = responses[responses$phase == phase & responses$data_index == 0,]
  #   View(firsts)
  counter = 1
  lengths <- vector()
  for(image in unique(firsts$image)) {
    for(practice in 0:1) {
      #         image = images[image_id,]
      #         print(image)
#       cat("image:", image, "phase:", phase, "practice:", practice, "\n")
      indices = which(responses$image == image & responses$is_practice == practice & responses$phase == phase & responses$data_index == 0)
      #         cat("Indices:", indices)
      first = responses[indices,'X']
      if (length(first) == 0) {
        print(paste("No responses for", image, "at phase", phase, "(practice?", practice, ")"))
        next
      }
      nxt = firsts[firsts$X > first,]
#       print(nxt[1,"X"])
#       cat("Length of next:", nrow(nxt), "\n")
      #       exit()
      if(nrow(nxt) < 1) {
#         print("Zero length..................")
        nxt = tail(responses)
      }
#       cat("First index:", first, "\n")
#       cat("Last index:", nxt[1,"X"], "\n")
      #         print(paste("Interval: ", print(first:(nxt-1))))
      traj = responses[responses$X %in% first:(nxt[1,"X"]-1),take_vars]
      if(is.null(d)) {
        d <- traj
      }
      else {
        d <- rbind(d, traj)  
      }
      #         View(traj)
#       d[[counter]] <- traj
#       print(head(dd))
#       print(nxt[1,"X"]-first)
      lengths[[counter]] <- nxt[1,"X"]-first
      cat("Trajectory",counter, " of length", nxt[1,"X"]-first, "processed.", "\n")
      counter = counter + 1
    }
  }
#   summary(d)
  if (length(take_vars) > 1) {
    d$x <- as.numeric(d$x)
    d$y <- as.numeric(d$y)
  }
  return(list(as.data.frame(d), lengths))
}

prepareData <- function(responses, phase) {
  d <- list()
  firsts = responses[responses$phase == phase & responses$data_index == 0,]
  #   View(firsts)
  counter = 1
  for(image in unique(firsts$image)) {
    for(practice in 0:1) {
      #         image = images[image_id,]
      #         print(image)
      cat("image:", image, "phase:", phase, "practice:", practice, "\n")
      indices = which(responses$image == image & responses$is_practice == practice & responses$phase == phase & responses$data_index == 0)
      #         cat("Indices:", indices)
      first = responses[indices,'X']
      if (length(first) == 0) {
        print(paste("No responses for", image, "at phase", phase, "(practice?", practice, ")"))
        next
      }
      nxt = firsts[firsts$X > first,]
      print(nxt[1,"X"])
      cat("Length of next:", nrow(nxt), "\n")
      #       exit()
      if(nrow(nxt) < 1) {
        print("Zero length..................")
        nxt = tail(responses)
      }
      cat("First index:", first, "\n")
      cat("Last index:", nxt[1,"X"], "\n")
      #         print(paste("Interval: ", print(first:(nxt-1))))
      traj = responses[responses$X %in% first:(nxt[1,"X"]-1),]
      selected <- data.matrix(traj)
      #         View(traj)
      d[[counter]] <- selected
      cat("Trajectory",counter, "processed.", "\n")
      counter = counter + 1
    }
  }
  return(d)
}

fitHMMtoPhase <- function(file_id, nStates, iter=100, dis="NORMAL", phase=NULL, nMixt=-1, take_vars=c('x','y')) {
  
  # This is a sample analysis thing using RHmm
  library("RHmm")
  source("~/Dropbox/ABACUS/Workspace/LeapArticulator/Data Analysis.R")
  
  list[responses, tests, images] <- readLogFile(file_id)
  d <- prepareData(responses, phase=phase)
#   setwd("~/Dropbox/ABACUS/Workspace/LeapArticulator/")
  # print(images)
  
#   print(summary(d))
  # fit HMM
  #  d = d[-(which(sapply(d,is.null),arr.ind=TRUE))]
  if (nMixt != -1) {
    # print("Mixture model")
    hmm <- HMMFit(d, nStates=nStates, dis=dis, asymptCov=FALSE, nMixt=nMixt, control=list(verbose=0, iter=iter))
    path = list()
    for(trajectory in d){
      path = c(path, viterbi(hmm, trajectory))
    }
  }
  else {
    # print("Normal model")
    # print(summary(d))
    hmm <- HMMFit(d, nStates=nStates, dis=dis, asymptCov=FALSE, control=list(verbose=0, iter=iter))
    #print("Trained!")
#     path = list()
#     for(trajectory in d){
#       path = c(path, viterbi(hmm, trajectory))
#     }
#     variances = hmm$HMM$distribution$cov
#     transmat = hmm$HMM$transMat
#     means = hmm$HMM$distribution$mean
  }
  
  return(list(hmm, d))
}

fitHMM <- function(file_id, nStates, iter=100, dis="NORMAL", up_to_phase=2, nMixt=-1) {
  
  # This is a sample analysis thing using RHmm
  setwd("~/ThereminData")
  library("RHmm")
  source("~/Dropbox/ABACUS/Workspace/LeapArticulator/Data Analysis.R")
  
  list[responses, tests, images] <- readLogFile(file_id)
  # print(images)
  d <- list()
  counter = 1
  for(image in images) {
    for(phase in 0:up_to_phase) {
      for(practice in 0:1) {
        # select responses to a particular image
        selected <- responses[responses$image == image & responses$phase == phase & responses$is_practice == practice,c('x','y')]
        selected <- data.matrix(selected)
        d[[counter]] <- selected
        counter = counter + 1
      }
    }
  }
  # fit HMM
  #  d = d[-(which(sapply(d,is.null),arr.ind=TRUE))]
  if (nMixt != -1) {
    # print("Mixture model")
    hmm <- HMMFit(d, nStates=nStates, dis=dis, asymptCov=FALSE, nMixt=nMixt, control=list(verbose=0, iter=iter))
    path = list()
    for(trajectory in d){
      path = c(path, viterbi(hmm, trajectory))
    }
  }
  else {
    # print("Normal model")
    # print(summary(d))
    hmm <- HMMFit(d, nStates=nStates, asymptCov=FALSE,dis=dis, control=list(verbose=0, iter=iter))
    #print("Trained!")
    path = list()
    for(trajectory in d){
      path = c(path, viterbi(hmm, trajectory))
    }
    variances = hmm$HMM$distribution$cov
    transmat = hmm$HMM$transMat
    means = hmm$HMM$distribution$mean
  }
  
  return(list(hmm, d))
}