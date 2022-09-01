write.csv(time_correlation_data, 
          "/Users/authors/repositories/Projects/complexity-verification-project/test.csv"
          , row.names = FALSE)

#----------------

correlation_data = time_correlation_data
meta_analysis_result_temp <- metacor(cor, n, data = correlation_data,
                                studlab = correlation_data$var_name,
                                sm = "ZCOR", comb.fixed=FALSE,
                                method.tau = "SJ")
#method.tau = method used to estimate the between-study variance \tau^2 and its square root \tau. 
#   "SJ" = Sidik-Jonkman estimator (method.tau = "SJ")
# See https://stats.stackexchange.com/questions/517779/how-to-choose-between-different-estimators-for-tau-in-meta-analysis

# The  Random-Effects-Model: https://cjvanlissa.github.io/Doing-Meta-Analysis-in-R/random.html
# Defaults settings: https://rdrr.io/cran/meta/man/settings.meta.html

#sm	=  summary measure ("ZCOR" or "COR") to be used for pooling of studies. (see Cooper et al., p264-5 and p273-4). 
#   ZCOR = Fisher's z transformation of correlations 
#   COR =  direct combination of (untransformed) correlations

# Only few statisticians would advocate the use of untransformed correlations unless sample sizes are very large (see Cooper et al., p265).

# Cooper H, Hedges LV, Valentine JC (2009): The Handbook of Research Synthesis and Meta-Analysis, 2nd Edition. New York: Russell Sage Foundation

# metanalysis with spearman? https://stats.stackexchange.com/questions/316093/meta-analysis-of-spearman-correlation-coefficients

#Michael Borenstein, Larry V Hedges, Julian PT Higgins, and Hannah R Rothstein. 2009. Effect Sizes Based on Correlations. In Introduction to meta-analysis. John Wiley & Sons, Chichester, United Kingdom, 40–43. 
#Michael Borenstein, Larry V Hedges, Julian PT Higgins, and Hannah R Rothstein. 2009. Random-Effects Model. In Introduction to meta-analysis. John Wiley & Sons, Chichester, United Kingdom, 68–75. 
#Michael Borenstein, Larry V Hedges, Julian PT Higgins, and Hannah R Rothstein. 2009. When does it make sense to perform a meta-analysis. In Introduction to meta-analysis. John Wiley & Sons, Chichester, United Kingdom, 357–64.

#Fixed-Effect Versus Random-Effects Models (Borenstein's book):
#   If the number of studies is very small, then theestimate of the between-studies variance (\tau^2) will have poor precision.

#The parameter tau-squared (\tau^2) is defined as the variance of the true effect sizes. Inother words, if we had an infinitely large sample of studies, each, itself, infinitelylarge (so that the estimate in each study was the true effect) and computed thevariance of these effects, this variance would be \tau^2.

#The statisticsT2(andT) reflect theamountoftrue heterogeneity (the variance or the standard deviation) whileI2reflects theproportion of observed dispersion that is due to this heterogeneity. In a sense, if wewere to multiply the observed variance byI2, we would getT2(this is meant as anillustration only, since the actual computation is more complicated). As such, thetwo tend to move in tandem, but have very different meanings

#In sum, when the number of studies is small, there are no really good options. As astarting point we would suggest reporting the usual statistics and then explaining thelimitations as clearly as possible. 

#a confidence interval, reflecting the precision with which the effect size has been estimatedin that study.

# Effect sizeOn the plot the summary effect is shown on the bottom line. In this example thesummary risk ratio is 0.85, indicating that the risk of death (or MI) was 15% lowerfor patients assigned to the high dose than for patients assigned to standard dose.The summary effect is nothing more than the weighted mean of the individualeffects.

# we cannot compute regular average because: (1) the sample size of studies is different, 
# (2) the random effects model indicates that each study has its own true effect size, 
#so we want to estimate the mean of the distribution of these sizes. 
#In this scenario, the average is not appropriate because it does not consider 
#the variance of the effect sizes of the individual studies
# (3) it does not consider the precisions of the studies. 
# We want to assign more weight to the more precise studies,

# dispersion of effect sizes 

#kendalls: https://towardsdatascience.com/kendall-rank-correlation-explained-dee01d99c535
# correlations cannot be averaged: https://medium.com/@jan.seifert/averaging-correlations-part-i-3adab6995042

#Andrew Gilpin (1993) has a nice table of conversions for tau to Pearson r, spearman rho, and common conversions thereof. I think this will suffice for your project.
#Gilpin, A. (1993). Table for conversion of Kendall's tau to Spearman's rho within the context of measures of magnitude of effect for meta-analysis. Educational and Psychological Measurement, 53, 87-92. doi:10.1177/0013164493053001007
#JMASM9: Converting Kendall’s Tau For Correlational Or Meta-Analytic Analyses

print(meta_analysis_result_temp)

#---------------


path = "/Users/authors/repositories/Projects/complexity-verification-project/"
dir.create(path, showWarnings = FALSE) # Create directory if it doesn't exist
png(file = paste(path, "test", "_forestplot.png", sep = ""), width = 1235, height = 575, res = 180)
pdf(file = paste(path, "test", "_forestplot.pdf", sep = ""))
forest_plot <- forest(meta_analysis_result_temp)
dev.off()
print(forest_plot)

#---------------


kendall_time_correlation_data2 <- print_correlation(time_vars, time_data, "kendall")