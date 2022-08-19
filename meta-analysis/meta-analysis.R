# NOTE: make sure to set your working directory first with setwd("Path/to/this/directory")

##---------------------------------------------------------------##
##                Load libraries and import data                 ##
##---------------------------------------------------------------##

#setwd("meta-analysis")

#delete all variables
rm(list = ls())

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
print_meta_analysis <- function(data_in, generateForestPlot, metric_type_in, sheet_in) {
  
  #filter by metric type and select the columns needed
  correlation_data = select(subset(data_in, metric_type == metric_type_in), 
                            c('dataset_metric', 
                               "num_snippets_for_correlation",
                               "pearson_r",
                               "kendalls_p_value"))
  #run the meta analysis
  meta_analysis_result <- metacor(pearson_r, num_snippets_for_correlation, 
                                  data = correlation_data,
                                  studlab = correlation_data$dataset_metric,
                                  sm = "ZCOR", comb.fixed=FALSE,
                                  method.tau = "SJ")
  print(meta_analysis_result)
  if (generateForestPlot) {
    path = "./forest-plot/"
    dir.create(path, showWarnings = FALSE) # Create directory if it doesn't exist
    png(file = paste(path, sheet_in, "_", metric_type_in, "_forestplot.png", sep = ""), 
        width = 1535, 
        height = 575, res = 180)
    #pdf(file = paste(path, metric_type_in, "_forestplot.pdf", sep = ""))
    forest_plot <- forest(meta_analysis_result, 
                          #prediction = TRUE, 
                          #smlab = "Correlation meta-analysis",
                          #smlab = "",
                          leftlabs = c("Metric", "# of snippets"))
    dev.off()
    print(forest_plot)
  }
}

#-----------------------------------

#run the meta-analysis on the data found in sheet_in
run_meta_analysis <- function(data_file_in, sheet_in){
  
  #read data
  all_data = read_excel(data_file_in, sheet = sheet_in)
  
  #select columns that I need
  all_data2 = select(all_data, c('dataset_id', 'metric', 'metric_type', 
                                 #'higher_warnings',
                                 "num_snippets_for_correlation",
                                 "kendalls_tau",
                                 "kendalls_p_value"))
  
  #rename columns
  # colnames(all_data2) <- c('dataset_id','metric_type',
  #                          'higher_warnings', 
  #                          "n",
  #                          "cor_tau",
  #                          "p_value")
  
  #concatenate the dataset ID and the metric
  all_data2$dataset_metric = paste(all_data2$dataset_id, "_", all_data2$metric, sep = "")
  
  #transform Kendall's tau into Pearson's r (cor)
  all_data2$pearson_r <- sapply(all_data2$kendalls_tau, convert_to_pearson)
  
  #get all the metric types
  metric_types = unique(all_data2$metric_type)
  
  #run the metanalysis for all the metric types
  lapply(metric_types, function (metric_type){print_meta_analysis(all_data2, TRUE, metric_type, sheet_in)})
}

#data_file = "correlation_analysis_for_meta_analysis.xlsx"
data_file = "../data/correlation_analysis.xlsx"
sheets = c("all_tools", "checker_framework", "typestate_checker", "infer")
lapply(sheets, function(sheet_in){run_meta_analysis(data_file, sheet_in)})

  