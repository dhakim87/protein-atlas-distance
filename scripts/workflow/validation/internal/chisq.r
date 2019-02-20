curve( dchisq(x, df=899), col='red', main = "Chi-Square Density Graph", from=0,to=10000)
xvec <- seq(1,10000,length=1000)
pvec <- dchisq(xvec,df=899)
polygon(c(xvec,rev(xvec)),c(pvec,rep(0,length(pvec))),
        col=adjustcolor("black",alpha=0.3))
