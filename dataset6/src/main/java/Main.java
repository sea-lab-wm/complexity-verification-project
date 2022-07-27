import weka.estimators.CheckEstimator;
import weka.estimators.EstimatorUtils;
import weka.gui.beans.ClassifierPerformanceEvaluatorCustomizer;
import weka.gui.beans.ModelPerformanceChart;
import weka.gui.experiment.GeneratorPropertyIteratorPanel;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello World");

        //ADDED BY KOBI
        Antlr4Master antlr4Master = new Antlr4Master();
        antlr4Master.runAll();

        CarReport carReport = new CarReport();
        carReport.runAll();

        HibernateORM hibernateORM = new HibernateORM();
        hibernateORM.runAll();

        K9 k9 = new K9();
        k9.runAll();

        MyExpenses myExpenses = new MyExpenses();
        myExpenses.runAll();

        OpenCMSCore openCMSCore = new OpenCMSCore();
        openCMSCore.runAll();

        Phoenix phoenix = new Phoenix();
        phoenix.runAll();

        //TEMP SKIPPED POM

        SpringBatch springBatch = new SpringBatch();
        springBatch.runAll();

        weka.estimators.CheckEstimator checkEstimator = new CheckEstimator();
        checkEstimator.runAll();

        weka.estimators.EstimatorUtils estimatorUtils = new EstimatorUtils();
        estimatorUtils.runAll();

        weka.gui.beans.ClassifierPerformanceEvaluatorCustomizer classifierPerformanceEvaluatorCustomizer = new ClassifierPerformanceEvaluatorCustomizer();
        classifierPerformanceEvaluatorCustomizer.runAll();

        weka.gui.beans.ModelPerformanceChart modelPerformanceChart = new ModelPerformanceChart();
        modelPerformanceChart.runAll();

        weka.gui.experiment.GeneratorPropertyIteratorPanel generatorPropertyIteratorPanel = new GeneratorPropertyIteratorPanel();
        generatorPropertyIteratorPanel.runAll();
    }
}