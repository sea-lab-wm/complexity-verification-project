# NOTE: make sure to set your working directory first with setwd("Path/to/this/directory")

##---------------------------------------------------------------##
##                Load libraries and import data                 ##
##---------------------------------------------------------------##

# Load libraries
library(readxl)
library(writexl)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(meta)

# Returns pearson's r for the given tau using Kendall's formula (1970)
convert_to_pearson <- function(tau) {
  return(sin(0.5 * pi * tau))
}


# Performs a correlation meta analysis on the given data and saves the forestplot to file
print_meta_analysis <- function(data_in, generateForestPlot, label) {
  
  #filter by label (i.e., metric type) and select the columns needed
  correlation_data = select(subset(data_in, metric_type == label), c('var_name', 
                                                                 "n",
                                                                 "cor_r",
                                                                 "p_value"))
  #run the meta analysis
  meta_analysis_result <- metacor(cor_r, n, data = correlation_data,
                                  studlab = correlation_data$var_name,
                                  sm = "ZCOR", comb.fixed=FALSE,
                                  method.tau = "SJ")
  print(meta_analysis_result)
  if (generateForestPlot) {
    path = "./forest-plot/"
    dir.create(path, showWarnings = FALSE) # Create directory if it doesn't exist
    png(file = paste(path, label, "_forestplot.png", sep = ""), 
        width = 1535, 
        height = 575, res = 180)
    #pdf(file = paste(path, label, "_forestplot.pdf", sep = ""))
    forest_plot <- forest(meta_analysis_result, 
                          #prediction = TRUE, 
                          #smlab = "Correlation meta-analysis",
                          #smlab = "",
                          leftlabs = c("Study-metric", "# of snippets"))
    dev.off()
    print(forest_plot)
  }
}

#-----------------------------------

#read data
all_tools_data = read_excel("correlation_analysis.xlsx", sheet = "all_tools")

#select columns that I need
all_tools_data2 = select(all_tools_data, c('dataset','metric_type',
                                           'higher_warnings', 
                                           "# of data points for correlation",
                                           "Kendall's Tau",
                                           "Kendall's p-value"))
#rename columns
colnames(all_tools_data2) <- c('dataset','metric_type',
                               'higher_warnings', 
                               "n",
                               "cor_tau",
                               "p_value")

#concatenate the dataset ID and the metric type
all_tools_data2$var_name = paste(all_tools_data2$dataset, "_", all_tools_data2$metric_type, sep = "")

#transform Kendall's tau into Pearson's r (cor)
all_tools_data2$cor_r <- sapply(all_tools_data2$cor_tau, convert_to_pearson)

#get all the metric types
metric_types = unique(all_tools_data2$metric_type)

#run the metanalysis for all the metric types
lapply(metric_types, function (label){print_meta_analysis(all_tools_data2, TRUE, label)})
  