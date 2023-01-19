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


print_meta_analysis_overall <- function(data_in, sheet_in){

  #filter by expected correlation
  correlation_data_pos = select(subset(data_in, expected_cor == "positive"), 
                            c('dataset_metric', 
                               "num_snippets_for_correlation",
                               "pearson_r",
                               "kendalls_p_value"))
  correlation_data_neg = select(subset(data_in, expected_cor == "negative"), 
                            c('dataset_metric', 
                               "num_snippets_for_correlation",
                               "pearson_r",
                               "kendalls_p_value"))

  file_name = paste(sheet_in, "_positive", sep = "")
  print_meta_analysis_generic(correlation_data_pos, file_name, 2.8)

  file_name = paste(sheet_in, "_negative", sep = "")
  print_meta_analysis_generic(correlation_data_neg, file_name, 4.3)

  #flip the correlation scores of the positve instances
  correlation_data = correlation_data_pos
  correlation_data$pearson_r = correlation_data$pearson_r*-1
  correlation_data$dataset_metric = paste(correlation_data$dataset_metric, " (+)", sep = "")
  correlation_data = rbind(correlation_data_neg, correlation_data)
  attach(correlation_data)
  correlation_data = correlation_data[order(dataset_metric),]
  detach(correlation_data)

  file_name = paste(sheet_in, "_positive_negated", sep = "")
  print_meta_analysis_generic(correlation_data, file_name, 5.5)

  #flip the correlation scores of the negative instances
  correlation_data = correlation_data_neg
  correlation_data$dataset_metric = paste(correlation_data$dataset_metric, " (-)", sep = "")
  correlation_data$pearson_r = correlation_data$pearson_r*-1
  correlation_data = rbind(correlation_data_pos, correlation_data)
  attach(correlation_data)
  correlation_data = correlation_data[order(dataset_metric),]
  detach(correlation_data)

  file_name = paste(sheet_in, "_negative_negated", sep = "")
  print_meta_analysis_generic(correlation_data, file_name, 5.5)
}

# Performs a correlation meta analysis on the given data and saves the forestplot to file
print_meta_analysis_generic <- function(correlation_data, forest_plot_file_name, chart_height=2.5) {

  #run the meta analysis
  meta_analysis_result <- metacor(pearson_r, num_snippets_for_correlation, 
                                  data = correlation_data,
                                  studlab = correlation_data$dataset_metric,
                                  sm = "ZCOR", comb.fixed=FALSE,
                                  method.tau = "SJ")
  print(meta_analysis_result)
  path = "./forest-plot/"
  dir.create(path, showWarnings = FALSE) # Create directory if it doesn't exist
  # png(file = paste(path, forest_plot_file_name, ".png", sep = ""), 
  #     width = 1535, 
  #     height = 575, res = 180)
  pdf(file = paste(path, forest_plot_file_name, ".pdf", sep = "")
      , 
      width = 8, 
      height = chart_height
      )
  forest_plot <- forest(meta_analysis_result, 
                        leftlabs = c("DS_Metric", "Snippets"),
                        rightlabs = c("r value", "95% CI   ", "Weight")
                        )
  dev.off()
  print(forest_plot)
}


# Performs a correlation meta analysis on the given data and saves the forestplot to file
print_meta_analysis <- function(data_in, generateForestPlot, metric_type_in, expected_cor_in, sheet_in) {

  #filter by metric type and select the columns needed
  correlation_data = select(subset(data_in, metric_type == metric_type_in & expected_cor == expected_cor_in), 
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
    # png(file = paste(path, sheet_in, "_", metric_type_in, "_", expected_cor_in, ".png", sep = ""), 
    #     width = 1535, 
    #     height = 575, res = 180)
    pdf(file = paste(path, sheet_in, "_", metric_type_in, "_", expected_cor_in, ".pdf", sep = "")
        , 
        width = 8, 
        height = 2.5
        #, res = 180
        )
    #pdf(file = paste(path, metric_type_in, "_forestplot.pdf", sep = ""))
    forest_plot <- forest(meta_analysis_result, 
                          #prediction = TRUE, 
                          #smlab = "Correlation meta-analysis",
                          #smlab = "",
                          leftlabs = c("DS_Metric", "Snippets"),
                          rightlabs = c("r value", "95% CI   ", "Weight")
                          )
    dev.off()
    #dev.off()
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
                                 "kendalls_p_value", 
                                 "expected_cor", "expected_cor"))                      
  all_data2 = subset(all_data2, !is.na(all_data2[,5])) 


  #keep only DS9 with no comments
  all_data2 = subset(all_data2, (dataset_id != "9_gc") & (dataset_id != "9_bc"))
  all_data2$dataset_id[all_data2$dataset_id == "9_nc"] <- "9"
  
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
  metric_types_expected_cor = unique(select(all_data2, c("metric_type", "expected_cor")))

  #run the metanalysis for all the metric types
  apply(metric_types_expected_cor, 1, function (metric_type_expected_cor){print_meta_analysis(all_data2, TRUE, metric_type_expected_cor[1], metric_type_expected_cor[2], sheet_in)})

  #run overall meta-analyses
  print_meta_analysis_overall(all_data2, sheet_in)
}

#data_file = "correlation_analysis_for_meta_analysis.xlsx"
data_file = "../data/correlation_analysis_timeout_max.xlsx"
sheets = c("all_tools", "checker_framework", "typestate_checker", "infer", "openjml")
#sheets = c("all_tools")
lapply(sheets, function(sheet_in){run_meta_analysis(data_file, sheet_in)})

  