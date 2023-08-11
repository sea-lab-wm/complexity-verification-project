package edu.wm.sealab.featureextraction;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Scanner;

public class Parser {
  public static void main(String[] args) {

    // String dirPath = args[1];
    String dirPath = "ml_model/src/main/resources/snippet_splitter_out/";
    // String dirPath = "ml_model/src/main/resources/raw_snippet_splitter_out/"; // uncomment for
    // raw features
    File projectDir = new File(dirPath);

    // Output features
    // File csvOutputFile = new File("ml_model/raw_feature_data.csv"); // uncomment for raw features
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
      pw.append("avgComments");
      pw.append("\n");

      Scanner scan = null;
      try {
        scan = new Scanner(new File("ml_model/loc_data.csv"));
        scan.useDelimiter("\n");
      } catch (FileNotFoundException e) {
        e.printStackTrace();
      }
      final Scanner sc = scan;
      sc.next();

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
                  JavaParser parser = new JavaParser(new ParserConfiguration().setAttributeComments(false));
                  cuNoComm = parser.parse(file).getResult().get();
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

                 String line;
                line = sc.next();
                String[] columns = line.split(",");
                double numLines = Double.parseDouble(columns[3]);
                double avgNumOfComments = features.getNumOfComments() / numLines;

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
                pw.append(new DecimalFormat("#.##").format(avgNumOfComments));
                pw.append("\n");
              })
          .explore(projectDir);
          sc.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }
}
