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

# this method prepares the data for running a meta analysis, and then runs it
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

# this is a naive implementation of a 3-level meta-analysis.
# it does a poor job of dealing with dependent effect sizes, so
# we don't use it. In particular, this implementation fails badly
# at assigning weights: it gives DS3 (the largest sample!) the smallest
# weight, because DS3 is the only DS with just one metric
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
# we chose not to use this approach, even though it gives somewhat more encouraging results,
# because we didn't fully understand it and we were concerned that we may have made a mistake.
# todo: figure out whether this is correct (maybe consult a statistician with experience in
# meta-analysis?)
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
  )
  
  return(meta_analysis_result)
}

# Performs a correlation meta analysis on the given data and saves the forestplot to file
print_meta_analysis_generic <- function(correlation_data, forest_plot_file_name, out_dir_in = ".", agg=TRUE) {

  print(paste("==================", forest_plot_file_name, "==================", sep=""))
  
  # TODO: if we change which meta-analysis method we're using, we need to make
  # a change here. Right now, all calls to this method use agg=TRUE.
  if (agg) {
    meta_analysis_result <- run_meta_analysis_agg(correlation_data)
  } else {
    meta_analysis_result <- run_meta_analysis_che(correlation_data)
  }
  
  print(summary(meta_analysis_result))
  
  print(predict(meta_analysis_result, transf=transf.ztor, digits=2))
  
  print("===============================")
  
  path = paste(out_dir_in, "../forest-plot/", sep = "")
  dir.create(path, showWarnings = FALSE) # Create directory if it doesn't exist

  pdf(file = paste(path, forest_plot_file_name, ".pdf", sep = "")
      ,
      width = 8,
      height = 5.5
      )
  par(mar=c(5,4,1,2))
  
  if (agg) {
    # these heights aren't perfect, but they're okay for non-publication-quality graphs
    # 9 is enough to fit 6 datasets; 23 is enough to fit all 20 metrics
    plot_ht <- 9
  } else {
    plot_ht <- 23
  } 
  
  forest_plot <- forest(meta_analysis_result, showweights=TRUE,
                        xlim=c(-5,5),
                        ylim=c(-2,plot_ht),
                        atransf=transf.ztor,
                        header="Dataset",
                        xlab="Pearson's r (negative correlation supports our hypothesis)",
                        ilab = cbind(num_snippets_for_correlation),
                        ilab.xpos= c(-3),
                        addpred=TRUE,
                        cex=0.75)
  text(c(-3,8),     meta_analysis_result$k+2, c("Number of Snippets"), cex=.75, font=2)
  text(c(3,8),     meta_analysis_result$k+2, c("Weights"), cex=.75, font=2)
  text(3.75, -1.5, bquote(paste("p = ", .(formatC(meta_analysis_result$pval, digits = 2, format = "f")), sep="")), cex=0.75, font=2)
  text(-3.1, -1.5, bquote(paste("Test for Heterogeneity: Q = ", .(formatC(meta_analysis_result$QE, digits=2, format="f")), 
                                ", df = ", .(meta_analysis_result$k - meta_analysis_result$p),
                                ", p = ", .(formatC(meta_analysis_result$QEp, digits=2, format="f")),
                                sep="")), cex=0.75, font=2)
                                              # "RE Model (Q = ",
                                              # .(formatC(meta_analysis_result$QE, digits=2, format="f")), ", df = ", .(meta_analysis_result$k - meta_analysis_result$p),
                                              # ", p = ", .(formatC(meta_analysis_result$QEp, digits=2, format="f")), "; ", I^2, " = ",
                                              # .(formatC(meta_analysis_result$I2, digits=1, format="f")), "%)")))
  dev.off()
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

  #keep only DS9 with no comments
  all_data2 = subset(all_data2, (dataset_id != "9_gc") & (dataset_id != "9_bc"))
  all_data2$dataset_id[all_data2$dataset_id == "9_nc"] <- "9"
  
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

data_file_all_tools = "../scatter_plots_timeout_max/all_tools_corr_data.csv"
data_file_infer = "../scatter_plots_timeout_max/infer_corr_data.csv"
data_file_checker_framework = "../scatter_plots_timeout_max/checker_framework_corr_data.csv"
data_file_typestate_checker = "../scatter_plots_timeout_max/typestate_checker_corr_data.csv"
data_file_openjml = "../scatter_plots_timeout_max/openjml_corr_data.csv"

data_file_no_infer = "../scatter_plots_ablation_timeout_max/no_infer_corr_data.csv"
data_file_no_checker_framework = "../scatter_plots_ablation_timeout_max/no_checker_framework_corr_data.csv"
data_file_no_typestate_checker = "../scatter_plots_ablation_timeout_max/no_typestate_checker_corr_data.csv"
data_file_no_openjml = "../scatter_plots_ablation_timeout_max/no_openjml_corr_data.csv"

run_meta_analysis(data_file_all_tools, "all_tools")
run_meta_analysis(data_file_infer, "infer")
run_meta_analysis(data_file_checker_framework, "checker_framework")
run_meta_analysis(data_file_typestate_checker, "typestate_checker")
run_meta_analysis(data_file_openjml, "openjml")

# ablation studies:
run_meta_analysis(data_file_no_infer, "no_infer")
run_meta_analysis(data_file_no_checker_framework, "no_checker_framework")
run_meta_analysis(data_file_no_typestate_checker, "no_typestate_checker")
run_meta_analysis(data_file_no_openjml, "no_openjml")
