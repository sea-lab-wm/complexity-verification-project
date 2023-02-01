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

# for aggregated meta-analysis
library(dmetar)

# for CHE meta-analysis
library(clubSandwich)

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
                              'metric_id',
                              'metric',
                              'fisher_z',
                              'fisher_z_sqrd_se'))
  correlation_data_neg = select(subset(data_in, expected_cor == "negative"),
                            c('dataset_metric',
                               "num_snippets_for_correlation",
                              'dataset_id',
                              'metric_id',
                              'metric',
                              'fisher_z',
                              'fisher_z_sqrd_se'))

  file_name = paste(sheet_in, "_positive", sep = "")
  #print_meta_analysis_generic(correlation_data_pos, file_name, 2.8, out_dir_in)

  # file_name = paste(sheet_in, "_negative", sep = "")
  #print_meta_analysis_generic(correlation_data_neg, file_name, 4.3, out_dir_in)

  if (nrow(correlation_data_pos) != 0) {
    #flip the correlation scores of the positive instances
    correlation_data = correlation_data_pos
    correlation_data$fisher_z = correlation_data$fisher_z*-1
    correlation_data$dataset_metric = paste(correlation_data$dataset_metric, " (+)", sep = "")
    correlation_data = rbind(correlation_data_neg, correlation_data)
    attach(correlation_data)
    correlation_data = correlation_data[order(dataset_metric),]
    detach(correlation_data)
  } else {
    # no positive expectations (e.g., for correctness metric type), so this is easy
    correlation_data = correlation_data_neg
  }
  
  file_name = paste(sheet_in, "_positive_negated_agg", sep = "")
  print_meta_analysis_generic(correlation_data, file_name, out_dir_in)

  #flip the correlation scores of the negative instances
  # correlation_data = correlation_data_neg
  # correlation_data$dataset_metric = paste(correlation_data$dataset_metric, " (-)", sep = "")
  # correlation_data$fisher_z = correlation_data$fisher_z*-1
  # correlation_data = rbind(correlation_data_pos, correlation_data)
  # attach(correlation_data)
  # correlation_data = correlation_data[order(dataset_metric),]
  # detach(correlation_data)
  # 
  # file_name = paste(sheet_in, "_negative_negated", sep = "")
  #print_meta_analysis_generic(correlation_data, file_name, 5.5, out_dir_in)
}

# uses the methodology in https://bookdown.org/MathiasHarrer/Doing_Meta_Analysis_in_R/es-calc.html#aggregate-es
# to combine studies
run_meta_analysis_agg <- function(correlation_data) {
  correlation_data_escalc = escalc(yi = fisher_z,           # Effect size
                                   sei = sqrt(fisher_z_sqrd_se),       # Standard error
                                   data = correlation_data)
  
  correlation_data.agg <- aggregate(correlation_data_escalc, 
                                    cluster = dataset_id,
                                    rho = 0.6)
  
  # print(correlation_data.agg)
  
  
  # run the meta analysis
  meta_analysis_result <- rma.mv(
    yi = fisher_z, # TODO: check name, should be the correlation column
    V = fisher_z_sqrd_se, # TODO: check name
    slab = dataset_id,
    data = correlation_data.agg,
    random = ~ 1 | dataset_id,
    test = "t",
    method = "REML"
  )
  return(meta_analysis_result)
}

run_meta_analysis_multi <- function(correlation_data) {
  #run the meta analysis
  meta_analysis_result <- rma.mv(
    yi = fisher_z, # TODO: check name, should be the correlation column
    V = fisher_z_sqrd_se, # TODO: check name
    slab = dataset_metric,
    data = correlation_data,
    random = ~ metric_id | dataset_id,
    test = "t",
    method = "REML"
  )
  return(meta_analysis_result)
}

# based on this https://bookdown.org/MathiasHarrer/Doing_Meta_Analysis_in_R/multilevel-ma.html#fit-rve
run_meta_analysis_che <- function(correlation_data) {
  
  # constant sampling correlation assumption
  rho <- 0.3 # TODO: if we use this model, we need to run a sensitivity analysis on rho
  # higher rho seems to result in more weight to the studies with more metrics;
  # low rho leads to more weight to the studies with bigger sample sizes.
  # the overall conclusions don't change that much either way, though.
  
  V <- with(correlation_data,
            impute_covariance_matrix(vi = fisher_z_sqrd_se,
                                     cluster = dataset_id,
                                     r = rho))
  
  #run the meta analysis
  meta_analysis_result <- rma.mv(
    yi = fisher_z, # TODO: check name, should be the correlation column
    V = V, # TODO: check name
    slab = dataset_metric,
    data = correlation_data,
    random = ~ 1 | metric_id,
    # sparse = TRUE
  )
  
  return(meta_analysis_result)
}

# Performs a correlation meta analysis on the given data and saves the forestplot to file
print_meta_analysis_generic <- function(correlation_data, forest_plot_file_name, out_dir_in = ".", agg=TRUE) {

  #print(correlation_data)
  if (agg) {
    meta_analysis_result <- run_meta_analysis_agg(correlation_data)
  } else {
    meta_analysis_result <- run_meta_analysis_che(correlation_data)
  }
  
  print(summary(meta_analysis_result))
  
  print(predict(meta_analysis_result, transf=transf.ztor, digits=2))
  
  path = paste(out_dir_in, "~/Research/complexity-verification/complexity-verification-project/forest-plot/", sep = "")
  dir.create(path, showWarnings = FALSE) # Create directory if it doesn't exist

  pdf(file = paste(path, forest_plot_file_name, ".pdf", sep = "")
      ,
      width = 8,
      height = 5.5
      )
  par(mar=c(5,4,1,2))
  
  if (agg) {
    plot_ht <- 9
  } else {
    plot_ht <- 23
  } 
  
  forest_plot <- forest(meta_analysis_result, showweights=TRUE,
                        xlim=c(-5,5),
                        ylim=c(-2,plot_ht),
                        cex=0.75)
  dev.off()
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
                                 "fisher_z",
                                 "fisher_z_sqrd_se",
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
  print_meta_analysis_generic(all_data2, no_tool_in, out_dir_in = out_dir)
}

#-----------------------------------

#run the meta-analysis on the data found in sheet_in
run_meta_analysis <- function(data_file_in, name){
  
  #read data
  all_data = read.csv(data_file_in)
  
  #select columns that I need
  all_data2 = select(all_data, c('dataset_id', 'metric', 'metric_type', 
                                 "num_snippets_for_correlation",
                                 "kendalls_tau",
                                 "kendalls_p_value",
				                         "fisher_z",
				                         "fisher_z_sqrd_se",				
                                 "expected_cor", "expected_cor"))  
  
  # remove metrics for datasets on which no considered tools
  # issue a warning
  all_data2 = subset(all_data2, kendalls_tau != "")
  
  print(all_data2)


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
  all_data2$metric_id <- 1:nrow(all_data2)
  
  #run the metanalysis for each metric type, if and only if we're considering all_tools
  if (name == "all_tools") {
    metric_types <- c("correctness", "rating", "time", "physiological")
    for (mt in metric_types) {
      mt_data <- subset(all_data2, metric_type == mt)
      print_meta_analysis_overall(mt_data, sheet_in = paste(name, "_", mt, sep = ""))
    }
  }
  
  #run overall meta-analyses
  print_meta_analysis_overall(all_data2, sheet_in = name)
}

data_file_all_tools = "~/Research/complexity-verification/complexity-verification-project/scatter_plots_timeout_max/all_tools_corr_data.csv"
data_file_infer = "~/Research/complexity-verification/complexity-verification-project/scatter_plots_timeout_max/infer_corr_data.csv"
data_file_checker_framework = "~/Research/complexity-verification/complexity-verification-project/scatter_plots_timeout_max/checker_framework_corr_data.csv"
data_file_typestate_checker = "~/Research/complexity-verification/complexity-verification-project/scatter_plots_timeout_max/typestate_checker_corr_data.csv"
data_file_openjml = "~/Research/complexity-verification/complexity-verification-project/scatter_plots_timeout_max/openjml_corr_data.csv"

run_meta_analysis(data_file_all_tools, "all_tools")
run_meta_analysis(data_file_infer, "infer")
run_meta_analysis(data_file_checker_framework, "checker_framework")
run_meta_analysis(data_file_typestate_checker, "typestate_checker")
run_meta_analysis(data_file_openjml, "openjml")

# ablation_data_folder = "../scatter_plots_ablation_timeout_max"
# tools = c("checker_framework", "typestate_checker", "infer", "openjml")
# lapply(tools, function(tool_in){run_ablation_meta_analysis(ablation_data_folder, tool_in)})