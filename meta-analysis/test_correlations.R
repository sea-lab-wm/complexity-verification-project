
raw_corr_data <- read.csv(here("~/Research/complexity-verification/complexity-verification-project//data/raw_correlation_data.csv"))

without_tools = select(subset(raw_corr_data, tool == "all_tools"), c('metric', 'snippet', 'metric_value', 'dataset'))

ds1 = subset(without_tools, dataset == "1")
ds2 = subset(without_tools, dataset == "2")
ds6 = subset(without_tools, dataset == "6")
ds9 = subset(without_tools, dataset == "9_nc")
dsf = subset(without_tools, dataset == "f")

ds1_rating=subset(ds1, metric=="correct_output_rating")
ds1_correctness=subset(ds1, metric=="output_difficulty")
ds1_time=subset(ds1, metric=="time_to_give_output")

cor(ds1_rating$metric_value, ds1_correctness$metric_value, method="kendall")
cor(ds1_rating$metric_value, ds1_time$metric_value, method="kendall")
cor(ds1_time$metric_value, ds1_correctness$metric_value, method="kendall")

ds2_p1=subset(ds2, metric=="brain_deact_31ant")
ds2_p2=subset(ds2, metric=="brain_deact_31post")
ds2_p3=subset(ds2, metric=="brain_deact_32")
ds2_time=subset(ds2, metric=="time_to_understand")

cor(ds2_p1$metric_value, ds2_p2$metric_value, method="kendall")
cor(ds2_p1$metric_value, ds2_p3$metric_value, method="kendall")
cor(ds2_p1$metric_value, ds2_time$metric_value, method="kendall")
cor(ds2_p2$metric_value, ds2_p3$metric_value, method="kendall")
cor(ds2_p2$metric_value, ds2_time$metric_value, method="kendall")
cor(ds2_p3$metric_value, ds2_time$metric_value, method="kendall")

ds6_rating=subset(ds6, metric=="binary_understandability")
ds6_correctness=subset(ds6, metric=="correct_verif_questions")
ds6_time=subset(ds6, metric=="time_to_understand")

cor(ds6_rating$metric_value, ds6_correctness$metric_value, method="kendall")
cor(ds6_rating$metric_value, ds6_time$metric_value, method="kendall")
cor(ds6_time$metric_value, ds6_correctness$metric_value, method="kendall")

ds9_gap=subset(ds9, metric=="gap_accuracy")
ds9_r1=subset(ds9, metric=="readability_level_before")
ds9_r2=subset(ds9, metric=="readability_level_ba")
ds9_time=subset(ds9, metric=="time_to_read_complete")

cor(ds9_gap$metric_value, ds9_r1$metric_value, method="kendall")
cor(ds9_gap$metric_value, ds9_r2$metric_value, method="kendall")
cor(ds9_gap$metric_value, ds9_time$metric_value, method="kendall")
cor(ds9_r1$metric_value, ds9_r2$metric_value, method="kendall")
cor(ds9_r1$metric_value, ds9_time$metric_value, method="kendall")
cor(ds9_r2$metric_value, ds9_time$metric_value, method="kendall")

dsf_p1=subset(dsf, metric=="brain_deact_31")
dsf_p2=subset(dsf, metric=="brain_deact_32")
dsf_percent=subset(dsf, metric=="perc_correct_output")
dsf_complex=subset(dsf, metric=="complexity_level")
dsf_time=subset(dsf, metric=="time_to_understand")

cor(dsf_p1$metric_value, dsf_p2$metric_value, method="kendall")
cor(dsf_p1$metric_value, dsf_percent$metric_value, method="kendall")
cor(dsf_p1$metric_value, dsf_time$metric_value, method="kendall")
cor(dsf_p1$metric_value, dsf_complex$metric_value, method="kendall")
cor(dsf_p2$metric_value, dsf_percent$metric_value, method="kendall")
cor(dsf_p2$metric_value, dsf_time$metric_value, method="kendall")
cor(dsf_p2$metric_value, dsf_complex$metric_value, method="kendall")
cor(dsf_percent$metric_value, dsf_time$metric_value, method="kendall")
cor(dsf_percent$metric_value, dsf_complex$metric_value, method="kendall")
cor(dsf_complex$metric_value, dsf_time$metric_value, method="kendall")
