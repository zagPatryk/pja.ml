#Author: Dominik Deja
#30.10.2021
#This code creates PDF file with 6 different scatter plots replicated multiple
# times with bootstrap sampling

# Poprawic opisy osi dla zmiennych kategorycznych

set.seed(2137)
setwd('C:/Users/zagpa/Desktop/Kodowanie/Data Science/pja/1/r')

pdf('TitanicBootstrap.pdf', width = 11, height = 8)
for(i in 1:3){
  
  data <- read.csv('../train.csv')
  data <- data[sample(1:dim(data)[1], replace=TRUE),]
  attach(data)
  plot(jitter(Pclass, 2), Age, pch=ifelse(Survived==1,1,4), 
       xlab="Ticket class", ylab="Age", main = "Titanic passengers",
       xaxt = 'n', axes=F, ylim=c(0,80))
  axis(1, at=1:3, labels=c("1st class", "2nd class", "3rd class"), lty=0)
  axis(2)
  for(j in c(1.5, 2.5)) abline(v=j, lty=2, col="grey")
  for(j in seq(from=0, to=80, by=5)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
  legend(par('usr')[2]-0.01*(par('usr')[2]-par('usr')[1]), 
         par('usr')[4]-4*par('usr')[3], 
         bty='n', xpd=NA, c("B"), pch=c(NA), text.col="grey")  

  plot(jitter(Age,20), Fare, pch=ifelse(Survived==1,1,4), xlab="Age", 
       ylab="Fare", xlim=c(0,80), ylim=c(0,100), main = "Titanic passengers")
  for(j in seq(from=0, to=80, by=5)) abline(v=j, lty=3, col="grey")
  for(j in seq(from=0, to=100, by=10)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
  legend(par('usr')[2]-0.01*(par('usr')[2]-par('usr')[1]), 
         par('usr')[4]-4*par('usr')[3], 
         bty='n', xpd=NA, c("B"), pch=c(NA), text.col="grey") 
    
  plot(jitter(Parch,2), jitter(SibSp,2), pch=ifelse(Survived==1,1,4), 
       xlab="#Parents and children aboard", ylab="#Siblings and spouses aboard", 
       main="Titanic passengers", axes=FALSE, xlim=c(-0.2,6.2), ylim=c(-0.2,8.1))
  axis(1, at=0:6, labels=0:6, lty=0)
  axis(2, at=0:8, labels=0:8, lty=0)
  for(j in seq(from=0.5, to=8.5, by=1)) abline(v=j, lty=2, col="grey")
  for(j in seq(from=0.5, to=8.5, by=1)) abline(h=j, lty=2, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
  legend(6.3875,#6.177, 
         9.76,#9.6, 
         bty='n', xpd=NA, c("B"), pch=c(NA), text.col="grey") 
    
  plot(jitter(ifelse(Sex=='male',0,1),1.9), Age, pch=ifelse(Survived==1,1,4), 
       xlab="Sex", ylab="Age", main = "Titanic passengers",
       xaxt="n", axes=FALSE, ylim=c(0,80))
  axis(1, at=0:1, labels=c("Male", "Female"), lty=0)
  axis(2)
  for(j in seq(from=0.5, to=1.5, by=1)) abline(v=j, lty=2, col="grey")
  for(j in seq(from=0, to=80, by=5)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
  legend(par('usr')[2]-0.01*(par('usr')[2]-par('usr')[1]), 
         par('usr')[4]-4*par('usr')[3], 
         bty='n', xpd=NA, c("B"), pch=c(NA), text.col="grey") 
    
  plot(jitter(ifelse(Sex=='male',0,1), 1.9), Fare, pch=ifelse(Survived==1,1,4), 
       xlab="Sex", ylab="Fare", ylim=c(0,100), 
       main = "Titanic passengers", axes=FALSE)
  axis(1, at=0:1, labels=c("Male", "Female"), lty=0)
  axis(2)
  for(j in seq(from=0.5, to=1.5, by=1)) abline(v=j, lty=2, col="grey")
  for(j in seq(from=0, to=100, by=10)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
  legend(par('usr')[2]-0.01*(par('usr')[2]-par('usr')[1]), 
         par('usr')[4]-4*par('usr')[3], 
         bty='n', xpd=NA, c("B"), pch=c(NA), text.col="grey") 
    
  plot(jitter(ifelse(Sex=='male',0,1), 1.9), Fare, pch=ifelse(Survived==1,1,4), 
       xlab="Sex", ylab=expression('Log'[10]*'(Fare)'),
       log="y", yaxt="n", main = "Titanic passengers", ylim=c(3,550), axes=FALSE)
  axis(1, at=0:1, labels=c("Male", "Female"), lty=0)
  for(j in seq(from=0.5, to=1.5, by=1)) abline(v=j, lty=2, col="grey")
  at.y <- outer(1:9, 10^(0:4))
  lab.y <- ifelse(log10(at.y) %% 1 == 0, at.y, NA)
  lab.y[5,] <- at.y[5,]
  axis(2, at=at.y, labels=lab.y, las=1)
  for(j in at.y) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         1.7375,
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
  legend(par('usr')[2]-0.01*(par('usr')[2]-par('usr')[1]), 
         1565, 
         bty='n', xpd=NA, c("B"), pch=c(NA), text.col="grey") 
    
  detach(data)
}
dev.off()

pdf('TitanicNoBootstrap.pdf', width = 11, height = 8)
for(i in 1:1){

  data <- read.csv('../train.csv')
  attach(data)
  plot(jitter(Pclass, 2), Age, pch=ifelse(Survived==1,1,4), 
       xlab="Ticket class", ylab="Age", main = "Titanic passengers",
       xaxt = 'n', axes=F, ylim=c(0,80))
  axis(1, at=1:3, labels=c("1st class", "2nd class", "3rd class"), lty=0)
  axis(2)
  for(j in c(1.5, 2.5)) abline(v=j, lty=2, col="grey")
  for(j in seq(from=0, to=80, by=5)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))

  plot(jitter(Age,20), Fare, pch=ifelse(Survived==1,1,4), xlab="Age", 
       ylab="Fare", xlim=c(0,80), ylim=c(0,100), main = "Titanic passengers")
  for(j in seq(from=0, to=80, by=5)) abline(v=j, lty=3, col="grey")
  for(j in seq(from=0, to=100, by=10)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))

  plot(jitter(Parch,2), jitter(SibSp,2), pch=ifelse(Survived==1,1,4), 
       xlab="#Parents and children aboard", ylab="#Siblings and spouses aboard", 
       main="Titanic passengers", axes=FALSE, xlim=c(-0.2,6.2), ylim=c(-0.2,8.1))
  axis(1, at=0:6, labels=0:6, lty=0)
  axis(2, at=0:8, labels=0:8, lty=0)
  for(j in seq(from=0.5, to=8.5, by=1)) abline(v=j, lty=2, col="grey")
  for(j in seq(from=0.5, to=8.5, by=1)) abline(h=j, lty=2, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
 
  plot(jitter(ifelse(Sex=='male',0,1),1.9), Age, pch=ifelse(Survived==1,1,4), 
       xlab="Sex", ylab="Age", main = "Titanic passengers",
       xaxt="n", axes=FALSE, ylim=c(0,80))
  axis(1, at=0:1, labels=c("Male", "Female"), lty=0)
  axis(2)
  for(j in seq(from=0.5, to=1.5, by=1)) abline(v=j, lty=2, col="grey")
  for(j in seq(from=0, to=80, by=5)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
 
  plot(jitter(ifelse(Sex=='male',0,1), 1.9), Fare, pch=ifelse(Survived==1,1,4), 
       xlab="Sex", ylab="Fare", ylim=c(0,100), 
       main = "Titanic passengers", axes=FALSE)
  axis(1, at=0:1, labels=c("Male", "Female"), lty=0)
  axis(2)
  for(j in seq(from=0.5, to=1.5, by=1)) abline(v=j, lty=2, col="grey")
  for(j in seq(from=0, to=100, by=10)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))

  plot(jitter(ifelse(Sex=='male',0,1), 1.9), Fare, pch=ifelse(Survived==1,1,4), 
       xlab="Sex", ylab=expression('Log'[10]*'(Fare)'),
       log="y", yaxt="n", main = "Titanic passengers", ylim=c(3,550), axes=FALSE)
  axis(1, at=0:1, labels=c("Male", "Female"), lty=0)
  for(j in seq(from=0.5, to=1.5, by=1)) abline(v=j, lty=2, col="grey")
  at.y <- outer(1:9, 10^(0:4))
  lab.y <- ifelse(log10(at.y) %% 1 == 0, at.y, NA)
  lab.y[5,] <- at.y[5,]
  axis(2, at=at.y, labels=lab.y, las=1)
  for(j in at.y) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         1.7375,
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
  
  detach(data)
}
dev.off()



pdf('TitanicAgeFare.pdf', width = 11, height = 8)
for(i in 1:30){
  
  data <- read.csv('../train.csv')
  data <- data[sample(1:dim(data)[1], replace=TRUE),]
  attach(data)
  
  plot(jitter(Age,20), Fare, pch=ifelse(Survived==1,1,4), xlab="Age", 
       ylab="Fare", xlim=c(0,80), ylim=c(0,100), main = "Titanic passengers")
  for(j in seq(from=0, to=80, by=5)) abline(v=j, lty=3, col="grey")
  for(j in seq(from=0, to=100, by=10)) abline(h=j, lty=3, col="grey")
  legend(par('usr')[2]-0.1*(par('usr')[2]-par('usr')[1]), 
         par('usr')[3]-0.06*(par('usr')[4]-par('usr')[3]), 
         bty='n', xpd=NA,
         c("Survived", "Deceased"), pch=c(1, 4))
  legend(par('usr')[2]-0.01*(par('usr')[2]-par('usr')[1]), 
         par('usr')[4]-4*par('usr')[3], 
         bty='n', xpd=NA, c("B"), pch=c(NA), text.col="grey") 

  detach(data)
}
dev.off()
