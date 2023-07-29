package edu.wm.sealab.featureextraction;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Parser {
  public static void main(String[] args) {

    // String dirPath = args[1];
    String dirPath = "ml_model/src/main/resources/snippet_splitter_out/";
    File projectDir = new File(dirPath);

    // Output features
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
      pw.append(",");
      pw.append("comments");
      pw.append("\n");

      new DirExplorer(
              (level, path, file) -> path.endsWith(".java"),
              (level, path, file) -> {
                // Compute the features on a single java file. The java file should contain a
                // single class surrounding a single method.
                FeatureVisitor featureVisitor = new FeatureVisitor();
                CompilationUnit cu = null;
                CompilationUnit cuNoComm = null;
                try {
                  cu = StaticJavaParser.parse(file);
                  StaticJavaParser.getParserConfiguration().setAttributeComments(false);
                  cuNoComm = StaticJavaParser.parse(file);
                } catch (FileNotFoundException e) {
                  e.printStackTrace();
                }
                featureVisitor.visit(cu, null);

                // Modify the CU to compute syntactic features i.e. parenthesis, commas, etc
                StringLiteralReplacer stringLiteralReplacer = new StringLiteralReplacer();
                stringLiteralReplacer.visit(cuNoComm, null);

                // Extract syntactic features (non JavaParser extraction)
                SyntacticFeatureExtractor sfe =
                    new SyntacticFeatureExtractor(featureVisitor.getFeatures());
                Features features = sfe.extract(cuNoComm.toString());

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
                pw.append(Integer.toString(features.getNumOfParameters()));
                pw.append(",");
                pw.append(Integer.toString(features.getNumOfIfStatements()));
                pw.append(",");
                pw.append(Integer.toString(features.getNumOfLoops()));
                pw.append(",");
                pw.append(Integer.toString(features.getAssignExprs()));
                pw.append(",");
                pw.append(Integer.toString(features.getCommas()));
                pw.append(",");
                pw.append(Integer.toString(features.getPeriods()));
                pw.append(",");
                pw.append(Integer.toString(features.getSpaces()));
                pw.append(",");
                pw.append(Integer.toString(features.getComparisons()));
                pw.append(",");
                pw.append(Integer.toString(features.getParenthesis()));
                pw.append(",");
                pw.append(Integer.toString(features.getLiterals()));
                pw.append(",");
                pw.append(Integer.toString(features.getNumOfComments()));
                pw.append("\n");
              })
          .explore(projectDir);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }
}
