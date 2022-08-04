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
  
  correlation_data = select(subset(data_in, metric_type == label), c('var_name', 
                                                                 "n",
                                                                 "cor",
                                                                 "p_value"))
  
  correlation_data$cor <- sapply(correlation_data$cor, convert_to_pearson)
  meta_analysis_result <- metacor(cor, n, data = correlation_data,
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


all_tools_data = read_excel("correlation_analysis.xlsx", sheet = "all_tools")

all_tools_data2 = select(all_tools_data, c('dataset','metric_type',
                                           'higher_warnings', 
                                           "# of data points for correlation",
                                           "Kendall's Tau",
                                           "Kendall's p-value"))
colnames(all_tools_data2) <- c('dataset','metric_type',
                               'higher_warnings', 
                               "n",
                               "cor",
                               "p_value")
all_tools_data2$var_name = paste(all_tools_data2$dataset, "_", all_tools_data2$metric_type, sep = "")

metric_types = unique(all_tools_data2$metric_type)

#print_meta_analysis(all_tools_data2, TRUE, metric_types[1])
lapply(metric_types, function (label){print_meta_analysis(all_tools_data2, TRUE, label)})
  
  
#sapply(all_tools_data2$cor, convert_to_pearson)