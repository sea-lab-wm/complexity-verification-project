write.csv(time_correlation_data, 
          "/Users/ojcchar/repositories/Projects/complexity-verification-project/test.csv"
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
print(meta_analysis_result_temp)

#---------------


path = "/Users/ojcchar/repositories/Projects/complexity-verification-project/"
dir.create(path, showWarnings = FALSE) # Create directory if it doesn't exist
png(file = paste(path, "test", "_forestplot.png", sep = ""), width = 1235, height = 575, res = 180)
pdf(file = paste(path, "test", "_forestplot.pdf", sep = ""))
forest_plot <- forest(meta_analysis_result_temp)
dev.off()
print(forest_plot)

#---------------


kendall_time_correlation_data2 <- print_correlation(time_vars, time_data, "kendall")