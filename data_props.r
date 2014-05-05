library(ggplot2)
library(scales)
library(rgl)

all<-read.csv('all-newlabels.csv')
all$label<-1*(all$label==0)

n_calc<-unlist(lapply(levels(factor(all$case)),function(x) sum(all$case==x)))
n_cac<-unlist(lapply(levels(factor(all$case)),function(x) sum((all$case==x)&(all$label==1))))
n_ncac<-unlist(lapply(levels(factor(all$case)),function(x) sum((all$case==x)&(all$label==0))))

df<-data.frame(levels(factor(all$case)),n_cac,n_ncac,n_calc)
names(df)<-c("case","n_cac","n_ncac","total")
ggplot(df,aes(x=total))+geom_histogram(binwidth=1,fill="firebrick3",col="white")+theme_bw()+xlab("# calcifications")+ylab("# patients")+ggtitle("Patients have varying numbers of calcifications")
ggsave("n_calc.pdf")

ggplot(subset(df,n_cac+n_ncac>1),aes(x=n_cac/(n_cac+n_ncac)))+geom_histogram(binwidth=0.05,fill="seagreen3",col="white")+theme_bw()+xlab("fraction of calcifications in coronary arteries")+ylab("# patients")+ggtitle("Patients have varying types of calcifications")
ggsave("frac_cac.pdf")

vars<-c("xl","yl","zl","mindist")
#vars<-c("vol","vsr","diml","dimh","xl","yl","zl","max","mean","kurt","skew","roll","pitch","yaw","mindist")
n<-nrow(all)
vartype<-c()
val<-c()
case<-rep(all$case,length(vars))
label<-rep(all$label,length(vars))
label<-ifelse(label==1,"CAC","non-CAC")
for (var in vars){
    vartype<-c(vartype,rep(var,n))
    val<-c(val,all[,var])
}
bigdf<-data.frame(val,vartype,label,case)
bigdf$label<-factor(label,c("non-CAC","CAC"))
bigdf$vartype<-factor(vartype,vars)

cc<-function(data,n){
    d<-subset(data,case==n)
    print(nrow(d))
    return(d)
}

dp<-function(data){
    plot3d(data$xl,data$yl,data$zl,col=ifelse(data$label==1,"red","grey"),xlim=c(0,1),ylim=c(0,1),zlim=c(0,1))
}

similar<-function(a,b){
    ret<-ifelse(a==b,ifelse(a==1,2,1),0)
    return(ret)
}

get_distance<-function(data){
    dist<-c()
    type<-c()
    space<-c(17,18,19)
    d<-as.matrix(dist(data[,space]))
    labs<-outer(data$label,data$label,similar)
    same_cac<-ifelse(labs==2,d,NA)
    same_nac<-ifelse(labs==1,d,NA)
    diff<-ifelse(labs==0,d,NA)
    same_cac<-same_cac[!is.na(same_cac)]
    same_nac<-same_nac[!is.na(same_nac)]
    diff<-diff[!is.na(diff)]
    dist<-c(same_cac,same_nac,diff)
    type<-c(rep("both cac",length(same_cac)),rep("both nac",length(same_nac)),rep("different",length(diff)))
    df<-data.frame(type,dist)
    df$type<-factor(type,c("both cac","both nac","different"))
    df<-df[df$dist!=0,]
    return(df)
}

ah<-get_distance(all)
ggplot(ah,aes(x=dist))+geom_histogram(binwidth=1,fill="firebrick3")+facet_grid(type~.,scales="free")+theme_bw()+xlab("Distance between calcification pair")+ggtitle("Distance between CACs demonstrates possible bimodality")
ggsave("dist_hist.pdf")
