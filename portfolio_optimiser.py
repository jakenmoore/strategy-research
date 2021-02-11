#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:04:54 2018

@author: jakemoore
"""

library(xts)
library(quantmod)
library(PerformanceAnalytics)
library(PortfolioAnalytics)

tickers <- c("AAPL", "MSFT", "AMZN")
start_date <- "2016-01-01"
percentile <- .95            # confidence level used in VaR calculations
 #
 # get Adjusted Close prices from Yahoo Finance
 # 
prices <- xts()
for(tick in tickers) {
        prices <- merge(prices, getSymbols(Symbols=tick, from=start_date, auto.assign=FALSE)[,paste(tick,"Adjusted",sep=".")])
        }
colnames(prices) <- tickers
#
# transform index from POSIXct to Date class
#
index(prices) <- as.Date(index(prices))
#
# compute returns
#  
testData_return <- diff(prices, arithmetic=FALSE, na.pad=FALSE) - 1
#
#  Compare VaR with quantile calculations for assets
#  when portfolio_method = "single" in VaR, the VaR for each column in R is calculated 
#
VaR_asset_hist <- VaR(R = testData_return, p=percentile, method="historical",
                        portfolio_method = "single")
print(VaR_asset_hist)
quant_asset_hist <- sapply(testData_return, quantile, probs=1-percentile, type=7)
print(quant_asset_hist)  
#
# Create the portfolio specification object 
#
Wcons <- portfolio.spec(assets = colnames(testData_return))
# 
# Add long_only and weight_sum = 1 constraints
#
Wcons <- add.constraint(portfolio = Wcons, type='box', min=0, max=1)  
Wcons <- add.constraint( portfolio=Wcons, type = "weight_sum",
                           min_sum=0.99, max_sum=1.01)          
#
# Set the objective to minimize VaR using historical returns
# portfolio_method ="component" tells VaR to use values of weights argument and calculate VaR for the portfolio
#
ObjSpec_hist = add.objective(portfolio = Wcons, type = "risk", 
                               name = "VaR", 
                               arguments=list(p=percentile, method="historical",
                                                            portfolio_method="component"),
                               enabled=TRUE)
opt <-  optimize.portfolio(R =testData_return, portfolio=ObjSpec_hist, 
                     search_size = 2000, trace = TRUE)
print(opt)
#
# compare VaR calculated using the optimization results with the quantile case.
# the VaR function calculates VaR slightly differently for historical data depending upon whether the 
# portfolio_method = "single" or "component".  The values for the quantile arguments probs and type used below should 
# give the same results for both the constrained_objective and quantile functions
#
VaR_port_opt <-constrained_objective(w=weights(opt),
                                       R=testData_return,portfolio = ObjSpec_hist)
quant_probs <- floor((1-percentile)*nrow(testData_return))/nrow(testData_return)
quant_port_opt <- quantile( testData_return%*%weights(opt), 
                              probs = quant_probs, type=1)
cat(paste("VaR using opt weights =", VaR_port_opt,
            "\nquantile calculation using opt weights =", quant_port_opt))