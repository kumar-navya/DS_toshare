trainLR.coef <- function(x){
    # return intercept and coefficients for train objects of  
    # penalized linear models with method value of lasso or ridge only
    
    # stop with error if object passed is not of class train.formula and 
    # is not called with method value of either lasso or ridge 
    
    if(!(class(x)[2] == "train.formula" & x$call$method %in% c("lasso", "ridge"))){
        stop("Not a train object with 'lasso' or 'ridge' method value")}
    
    tunevalue = numeric()
    
    # if object is a LASSO train.formula, tunevalue = bestTune$fraction
    # else it is a ridge train, hence tunevalue = 1
    # for choice of tunevalue, see predict method in x$modelInfo
    
    ifelse(x$call$method == "lasso", 
           tunevalue <- x$bestTune$fraction, 
           tunevalue <- 1
           )
    
    
    # intecept = Yhat - sum(Xs * coefficients of Xs)
    
    intercept = 
        as.numeric(predict(x, x$trainingData[1:2,]))[1] - 
        sum(x$trainingData[1,-1] * 
                predict(x$finalModel, 
                        type = "coefficients", 
                        mode = "fraction", 
                        s = tunevalue
                )$coefficients
            )
    
    # coefficients
    coefs = predict(x$finalModel, 
                    type = "coefficients", 
                    mode = "fraction", 
                    s = tunevalue
                    )$coefficients
    
    # concatenating intercept and coefficients 
    output = c(intercept, coefs)
    # named numeric output, 
    # with names being (Intercept) or names of the explanatory variables
    attributes(output)$names[1] = "(Intercept)"
    
    return(output)

}


trainLR.plot <- function(x, mode = "fraction"){
    
    # stop with error message of passed object is not of class train.formula
    # or of method value 'lasso' or 'ridge'
    if(!(class(x)[2] == "train.formula" & x$call$method %in% c("lasso", "ridge"))){
        stop("Not a train object with 'lasso' or 'ridge' method value")}
    
    # stop with error message if argument mode is other than 'fraction' or 'penalty'
    if(!(mode %in% c("fraction", "penalty"))){
        stop("Mode can take values: 'fraction' or 'penalty'")}
    
    # get all iterations of fractions of full solution and coefficient values
    # which have been evaluated before choosing best solution: delivering least RMSE
    model = predict(x$finalModel, type = "coefficients", mode = "fraction", 
                    s = predict(x$finalModel, 
                                type = "coefficients", 
                                mode = "fraction"
                                )$fraction
                    )
    
    # named vector of 'n' distinct colors
    colorlist = character()
    for(i in 1:ncol(model$coefficient)){
        
        colorlist[i] = rgb(sample(seq(0, 200, by = 1), 1), 
                           sample(seq(0, 200, by = 1), 1), 
                           sample(seq(0, 200, by = 1), 1), 
                           maxColorValue = 255)
        
    }
        
    coefnames = colnames(model$coefficients)
    attributes(colorlist)$names = coefnames
    
    # optimal value of tuning parameter
    tunevalue = numeric()
    ifelse(x$call$method == "lasso", 
           tunevalue <- x$bestTune$fraction, 
           tunevalue <- 1
    )
    
    xvals = numeric()
    xlims = numeric(); optitune = numeric(); labelpos = numeric()
    
    ifelse(mode == "fraction", 
           {# fraction
               xvals <- model$fraction
               xlims <- c(0, range(xvals)[2] + 0.2)
               optitune <- tunevalue
               varlabelpos <- 1.05
               tunepos <- 2
               plottitle <- "fraction of full solution"
           }, 
           {# penalty
               xvals <- x$finalModel$penalty
               xlims <- c(-10, range(xvals)[2])
               optitune <- xvals[which(model$fraction == tunevalue)]
               varlabelpos <- -3
               tunepos <- 4
               plottitle <- "penalty"
           }
    )
    
    plot(x = xvals, y = model$coefficients[,1], 
         type = "l", lwd = 1, 
         col = colorlist[coefnames[1]], 
         ylim = range(model$coefficients), 
         xlim = xlims, 
         xlab = mode, ylab = "coefficients")
    
    for(i in 2:ncol(model$coefficients)){
        points(x = xvals, y = model$coefficients[,i], 
               type = "l", lwd = 1, 
               col = colorlist[coefnames[i]])
        }
    
    abline(v = optitune, h = 0, lty = 2, col = "lightgrey")
    
    text(x = optitune, y = min(model$coefficients), 
         labels = paste(round(optitune,2), ": optimal",mode, "(least RMSE)"), cex = 0.8, pos = tunepos)
    
    for(i in 1:ncol(model$coefficients)){
        points(x = optitune, 
               y = model$coefficients[which(model$fraction == tunevalue),i], 
               col = colorlist[coefnames[i]], 
               pch = 20, cex = 1
               )
    }
    
    coeflastrow = sort(model$coefficients[nrow(model$coefficients),], decreasing = TRUE)
    
    text(x = varlabelpos, 
         y = coeflastrow, 
         labels = attributes(coeflastrow)$names, 
         col = colorlist[attributes(coeflastrow)$names]
         )
    
    title(main = paste(toupper(x$call$method),": coefficients vs. ", plottitle, sep = ""))
    
}