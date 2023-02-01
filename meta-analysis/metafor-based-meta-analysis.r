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

# removed in favor of metafor
# library(meta)

library(metafor)

# Returns pearson's r for the given tau using Kendall's formula (1970)
convert_to_pearson <- function(tau) {
  return(sin(0.5 * pi * tau))
}


print_meta_analysis_overall <- function(data_in, sheet_in="all_tools", out_dir_in = ""){

  #filter by expected correlation
  correlation_data_pos = select(subset(data_in, expected_cor == "positive"), 
                            c('dataset_metric',
                              "num_snippets_for_correlation",
                              'dataset_id',
                              'metric',
                              'fisher_z',
                              'fizher_z_sqrd_se'))
  correlation_data_neg = select(subset(data_in, expected_cor == "negative"), 
                            c('dataset_metric', 
                               "num_snippets_for_correlation",
                              'dataset_id',
                              'metric',
                              'fisher_z',
                              'fizher_z_sqrd_se'))

  file_name = paste(sheet_in, "_positive", sep = "")
  #print_meta_analysis_generic(correlation_data_pos, file_name, 2.8, out_dir_in)

  file_name = paste(sheet_in, "_negative", sep = "")
  #print_meta_analysis_generic(correlation_data_neg, file_name, 4.3, out_dir_in)

  #flip the correlation scores of the positive instances
  correlation_data = correlation_data_pos
  correlation_data$fisher_z = correlation_data$fisher_z*-1
  correlation_data$dataset_metric = paste(correlation_data$dataset_metric, " (+)", sep = "")
  correlation_data = rbind(correlation_data_neg, correlation_data)
  attach(correlation_data)
  correlation_data = correlation_data[order(dataset_metric),]
  detach(correlation_data)

  file_name = paste(sheet_in, "_positive_negated", sep = "")
  print_meta_analysis_generic(correlation_data, file_name, 5.5, out_dir_in)

  #flip the correlation scores of the negative instances
  correlation_data = correlation_data_neg
  correlation_data$dataset_metric = paste(correlation_data$dataset_metric, " (-)", sep = "")
  correlation_data$fisher_z = correlation_data$fisher_z*-1
  correlation_data = rbind(correlation_data_pos, correlation_data)
  attach(correlation_data)
  correlation_data = correlation_data[order(dataset_metric),]
  detach(correlation_data)

  file_name = paste(sheet_in, "_negative_negated", sep = "")
  #print_meta_analysis_generic(correlation_data, file_name, 5.5, out_dir_in)
}

# Performs a correlation meta analysis on the given data and saves the forestplot to file
print_meta_analysis_generic <- function(correlation_data, forest_plot_file_name, chart_height=2.5, out_dir_in = ".") {

  print(correlation_data)
  
  #run the meta analysis
  meta_analysis_result <- rma.mv(
    yi = fisher_z, # TODO: check name, should be the correlation column
    V = fizher_z_sqrd_se, # TODO: check name
    slab = dataset_metric,
    data = correlation_data,
    random = ~ 1 | dataset_id/metric,
    test = "t",
    method = "REML"
  )
  
  print(summary(meta_analysis_result))
  
  # print(meta_analysis_result)
  # if(out_dir_in == ""){
  #   out_dir_in = "."
  # }
  path = paste(out_dir_in, "~/Research/complexity-verification/complexity-verification-project/forest-plot/", sep = "")
  dir.create(path, showWarnings = FALSE) # Create directory if it doesn't exist
  # png(file = paste(path, forest_plot_file_name, ".png", sep = ""),
  #     width = 1535,
  #     height = 575, res = 180)
  pdf(file = paste(path, forest_plot_file_name, ".pdf", sep = "")
      ,
      width = 8,
      height = chart_height
      )
  par(mar=c(5,4,1,2))
  forest_plot <- forest(meta_analysis_result, showweights=TRUE,
                        xlim=c(-5,5),
                        ylim=c(-2,23),
                        cex=0.75)
  dev.off()
  # print(forest_plot)
}


# Performs a correlation meta analysis on the given data and saves the forestplot to file
print_meta_analysis <- function(data_in, generateForestPlot, metric_type_in, expected_cor_in, sheet_in) {

  #filter by metric type and select the columns needed
  correlation_data = select(subset(data_in, metric_type == metric_type_in & expected_cor == expected_cor_in), 
                            c('dataset_id', 'metric', 'fisher_z', 'fizher_z_sqrd_se'))

  #run the meta analysis
  # This version is wrong: it does not account for the Unit-of-Analysis problem.
  # See https://bookdown.org/MathiasHarrer/Doing_Meta_Analysis_in_R/effects.html#unit-of-analysis
  # for a description of the problem.
  #   meta_analysis_result <- metacor(pearson_r, num_snippets_for_correlation, 
  #                                  data = correlation_data,
  #                                studlab = correlation_data$dataset_metric,
  #                                sm = "ZCOR", comb.fixed=FALSE,
  #                                method.tau = "SJ")


  meta_analysis_result <- rma.mv(
  		       yi = fisher_z, # TODO: check name, should be the correlation column
		       V = fizher_z_sqrd_se, # TODO: check name
		       slab = dataset_id,
		       data = correlation_data,
		       random = ~ 1 | dataset_id/metric,
		       test = "t",
		       method = "REML"
  )

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

run_ablation_meta_analysis <- function(ablation_data_folder_in, tool_in){
  #read data from CSV file
  file_path = paste(ablation_data_folder_in, "/", "no_", tool_in, "_corr_data.csv", sep = "")
  all_data = read.csv(file_path)

  #set column names to lower case
  column_names = names(all_data)
  new_column_names = lapply(column_names,  tolower)
  colnames(all_data) <- new_column_names

  # TODO: update this to work with the new meta-analysis method

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

  #concatenate the dataset ID and the metric
  all_data2$dataset_metric = paste(all_data2$dataset_id, "_", all_data2$metric, sep = "")
  
  #transform Kendall's tau into Pearson's r (cor)
  all_data2$pearson_r <- sapply(all_data2$kendalls_tau, convert_to_pearson)

  #create output directory if it does not exit
  out_dir = "ablation"
  dir.create(out_dir, showWarnings = FALSE)

  #run overall meta-analyses
  no_tool_in = paste("no_", tool_in, sep = "")
  print_meta_analysis_overall(all_data2, no_tool_in, out_dir)
}

#-----------------------------------

#run the meta-analysis on the data found in sheet_in
run_meta_analysis <- function(data_file_in){
  
  #read data
  all_data = read.csv(data_file_in)
  
  #select columns that I need
  all_data2 = select(all_data, c('dataset_id', 'metric', 'metric_type', 
                                 #'higher_warnings',
                                 "num_snippets_for_correlation",
                                 "kendalls_tau",
                                 "kendalls_p_value",
				  "fisher_z",
				  "fizher_z_sqrd_se",				
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
  # apply(metric_types_expected_cor, 1, function (metric_type_expected_cor){print_meta_analysis(all_data2, TRUE, metric_type_expected_cor[1], metric_type_expected_cor[2], sheet_in)})

  #run overall meta-analyses
  print_meta_analysis_overall(all_data2)
}

#data_file = "correlation_analysis_for_meta_analysis.xlsx"
data_file = "~/Research/complexity-verification/complexity-verification-project/scatter_plots_timeout_max/all_tools_corr_data.csv"
# sheets = c("all_tools", "checker_framework", "typestate_checker", "infer", "openjml")
sheets = c("all_tools")

run_meta_analysis(data_file)

# ablation_data_folder = "../scatter_plots_ablation_timeout_max"
# tools = c("checker_framework", "typestate_checker", "infer", "openjml")
# lapply(tools, function(tool_in){run_ablation_meta_analysis(ablation_data_folder, tool_in)})