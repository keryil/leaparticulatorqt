
readLogFile <- function(id) {
  
  setwd("~/ThereminData")
  
  images <- unique(read.csv(file=paste("logs/", id, ".exp.images.csv", sep=""), sep="|"))
    
  responses <- read.csv(file=paste("logs/", id, ".exp.responses.freq_and_amp.csv", sep=""), sep="|")
  responses$is_practice <- factor(responses$is_practice)
  responses$phase <- factor(responses$phase)
  
  tests <- read.csv(file=paste("logs/", id, ".exp.tests.csv", sep=""), sep="|")
  tests$image0 = factor(tests$image0, levels = images$image_name)
  tests$image1 = factor(tests$image1, levels = images$image_name)
  tests$image2 = factor(tests$image2, levels = images$image_name)
  tests$image3 = factor(tests$image3, levels = images$image_name)
  tests$answer = factor(tests$answer, levels = images$image_name)
  tests$given_answer = factor(tests$given_answer, levels = images$image_name)
  
  tests$is_practice <- factor(tests$is_practice)
  tests$phase <- factor(tests$phase)
  tests$passed <- tests$answer == tests$given_answer
  return(list(responses, tests, images))
}

list <- structure(NA,class="result")
"[<-.result" <- function(x,...,value) {
  args <- as.list(match.call())
  args <- args[-c(1:2,length(args))]
  length(value) <- length(args)
  for(i in seq(along=args)) {
    a <- args[[i]]
    if(!missing(a)) eval.parent(substitute(a <- v,list(a=a,v=value[[i]])))
  }
  x
}