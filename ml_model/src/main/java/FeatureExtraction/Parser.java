package FeatureExtraction;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

import FeatureExtraction.FeatureExtractor.FeatureVisitor;
import FeatureExtraction.FeatureExtractor.StringLiteralReplacer;

public class Parser {
    public static void main(String[] args) {
        //Snippet splitting
        SnippetSplitter ss = new SnippetSplitter("ml_model/src/main/resources/manually_created_snippets/", "ml_model/src/main/resources/snippet_splitter_out/");
		ss.run(new File("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/"), "1", "SNIPPET_STARTS");
        ss.run(new File("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/"), "2", "DATASET2START");
        ss.run(new File("simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/"), "3", "SNIPPET_STARTS");
        ss.run(new File("dataset6/src/main/java/"), "6", "SNIPPET_STARTS");
        ss.run(new File("dataset9/src/main/java/"), "9$gc", "SNIPPET_STARTS_1");
        ss.run(new File("dataset9/src/main/java/"), "9$bc", "SNIPPET_STARTS_2");
        ss.run(new File("dataset9/src/main/java/"), "9$nc", "SNIPPET_STARTS_3");
        ss.run(new File("simple-datasets/src/main/java/fMRI_Study_Classes/"), "f", "SNIPPET_STARTS");

        //Parsing
        FeatureExtractor featureExtractor = new FeatureExtractor();
        
        //String dirPath = args[1];
        String dirPath = "ml_model/src/main/resources/snippet_splitter_out/";
        File projectDir = new File(dirPath);

        //Output features
        File csvOutputFile = new File("ml_model/feature_data.csv");
        try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
			// write header row
			pw.append("dataset_id");
			pw.append(",");
			pw.append("snippet_id");
			pw.append(",");
            pw.append("method_name");
			pw.append(",");
			pw.append("file");
			pw.append(",");
			pw.append("parameters");
			pw.append(",");
			pw.append("ifStatements");
			pw.append(",");
			pw.append("loops");
			pw.append(",");
			pw.append("assignmentExpressions");
			pw.append(",");
			pw.append("commas");
			pw.append(",");
			pw.append("periods");
			pw.append(",");
			pw.append("spaces");
			pw.append(",");
			pw.append("comparisons");
			pw.append(",");
			pw.append("parenthesis");
			pw.append(",");
			pw.append("literals");
			pw.append("\n");
			
			new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
			    // Compute the features on a single java file. The java file should contain a single class surrounding a single method.
			    FeatureVisitor featureVisitor = featureExtractor.new FeatureVisitor();
			    CompilationUnit cu = null;
				CompilationUnit cuNoComm = null;
			    try {
			        cu = StaticJavaParser.parse(file);
					StaticJavaParser.getParserConfiguration().setAttributeComments(false);
					cuNoComm = StaticJavaParser.parse(file);
			    } catch (FileNotFoundException e) {
			        e.printStackTrace();
			    }
			    featureVisitor.visit(cu,null);
				
				// Modify the CU to compute syntactic features i.e. parenthesis, commas, etc
				StringLiteralReplacer stringLiteralReplacer = featureExtractor.new StringLiteralReplacer();
				stringLiteralReplacer.visit(cuNoComm, null);

				// Extract syntactic features (non JavaParser extraction)
				SyntacticFeatureExtractor sfe = new SyntacticFeatureExtractor(featureVisitor.getFeatures());
				FeatureMap featureMap = sfe.extract(cuNoComm.toString());

			    // Add the extracted features to the CSV file
                String[] parts = file.getName().split("_");

			    pw.append(parts[1].replace("$", "_"));
			    pw.append(",");
			    pw.append(parts[3].replace("$", "-"));
			    pw.append(",");
                pw.append(parts[4].split("\\.")[0]);
			    pw.append(",");
			    pw.append(file.getName());
			    pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfParameters()));
			    pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfIfStatements()));
			    pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfLoops()));
				pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfAssignExprs()));
				pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfCommas()));
				pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfPeriods()));
				pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfSpaces()));
				pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfComparisons()));
				pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfParenthesis()));
				pw.append(",");
			    pw.append(Integer.toString(featureMap.getNumOfLiterals()));
			    pw.append("\n");
			}).explore(projectDir);
		} catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
