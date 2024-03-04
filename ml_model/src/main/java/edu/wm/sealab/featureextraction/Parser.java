package edu.wm.sealab.featureextraction;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class Parser {
  public static void main(String[] args) {
    //This file reads from loc_data.csv and outputs to feature_data.csv. It does not handle raw data.

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
      pw.append("statements");
      pw.append(",");
      pw.append("avgConditionals");
      pw.append(",");
      pw.append("avgLoops");
      pw.append(",");
      pw.append("avgAssignmentExpressions");
      pw.append(",");
      pw.append("avgComparisons");
      pw.append(",");
      pw.append("avgComments");
      pw.append(",");
      pw.append("avgConditionals");
      pw.append(",");
      pw.append("avgCommas");
      pw.append(",");
      pw.append("avgParenthesis");
      pw.append(",");
      pw.append("avgPeriods");
      pw.append(",");
      pw.append("avgSpaces");
      pw.append(",");
      pw.append("avgLineLength");
      pw.append(",");
      pw.append("maxLineLength");
      pw.append(",");
      pw.append("avgIndentation");
      pw.append(",");
      pw.append("maxIndentation");
      pw.append(",");
      pw.append("avgBlankLines");
      pw.append(",");
      pw.append("avgArithmeticOperators");
      pw.append(",");
      pw.append("avgNumbers");
      pw.append(",");
      pw.append("maxNumbers");
      pw.append(",");
      pw.append("keywordsX");
      pw.append(",");
      pw.append("identifiersX");
      pw.append(",");
      pw.append("operatorsX");
      pw.append(",");
      pw.append("numbersX");
      pw.append(",");
      pw.append("stringsX");
      pw.append(",");
      pw.append("literalsX");
      pw.append(",");
      pw.append("commentsX");
      pw.append(",");
      pw.append("keywordsY");
      pw.append(",");
      pw.append("identifiersY");
      pw.append(",");
      pw.append("operatorsY");
      pw.append(",");
      pw.append("numbersY");
      pw.append(",");
      pw.append("stringsY");
      pw.append(",");
      pw.append("literalsY");
      pw.append(",");
      pw.append("commentsY");
      pw.append("\n");

      List<String[]> lines = null;
      try (FileReader fileReader = new FileReader("ml_model/loc_data.csv");
          CSVReader csvReader = new CSVReaderBuilder(fileReader).withSkipLines(1).build(); ) {
        lines = csvReader.readAll();
      } catch (IOException e) {
        e.printStackTrace();
      }
      final List<String[]> allLines = lines;
      new DirExplorer(
              (level, path, file) -> path.endsWith(".java"),
              (level, path, file) -> {
                // Compute the features on a single java file. The java file should contain a
                // single class surrounding a single method.
                FeatureVisitor featureVisitor = new FeatureVisitor();
                VisualFeatureVisitor visualFeatureVisitor = new VisualFeatureVisitor();
                CompilationUnit cu = null;
                CompilationUnit cuNoComm = null;
                String[] split = null;
                try {
                  cu = StaticJavaParser.parse(file);
                  JavaParser parser =
                      new JavaParser(new ParserConfiguration().setAttributeComments(false));
                  cuNoComm = parser.parse(file).getResult().get();
                  split = Files.readString(file.toPath()).split("\n");
                } catch (IOException e) {
                  e.printStackTrace();
                }
                visualFeatureVisitor.getVisualFeatures().makeVisualFeaturesMatrix(split);

                featureVisitor.visit(cu, null);

                visualFeatureVisitor.visit(cu, null);
                VisualFeatures visualFeatures = visualFeatureVisitor.getVisualFeatures();
                visualFeatures.findVisualXY();

                // Modify the CU to compute syntactic features i.e. parenthesis, commas, etc
                StringLiteralReplacer stringLiteralReplacer = new StringLiteralReplacer();
                stringLiteralReplacer.visit(cuNoComm, null);

                // Extract syntactic features (non JavaParser extraction)
                SyntacticFeatureExtractor sfe =
                    new SyntacticFeatureExtractor(featureVisitor.getFeatures());

                String methodBody = cuNoComm.findFirst(MethodDeclaration.class)
                            .map(method -> method.toString())
                            .orElse("");
                Features features = sfe.extract(methodBody);
                
                // Locate and extract file data from loc_data.csv
                int entryIndex = findCorrespondingEntry(allLines, file.getName());
                String[] entryLine = allLines.get(entryIndex);
                double entryNumLinesOfCode = Double.parseDouble(entryLine[4]);

                allLines.remove(entryIndex);

                // Calculate averages based on data from loc_data.csv
                double avgNumOfComments = features.getNumOfComments() / entryNumLinesOfCode;
                double avgNumOfComparisons = features.getComparisons() / entryNumLinesOfCode;
                double avgNumOfArithmeticOperators =
                    features.getArithmeticOperators() / entryNumLinesOfCode;
                double avgNumOfConditionals = features.getConditionals() / entryNumLinesOfCode;
                
                double avgCommas = features.getCommas() / (features.getTotalBlankLines() + entryNumLinesOfCode);
                double avgParenthesis = features.getParenthesis() / (features.getTotalBlankLines() + entryNumLinesOfCode);
                double avgPeriods = features.getPeriods() / (features.getTotalBlankLines() + entryNumLinesOfCode);
                double avgSpaces = features.getSpaces() / (features.getTotalBlankLines() + entryNumLinesOfCode);
                double avgLineLength = features.getTotalLineLength() / (features.getTotalBlankLines() + entryNumLinesOfCode);
                double avgIndentationLength = features.getTotalIndentation() / (features.getTotalBlankLines() + entryNumLinesOfCode);
                double avgBlankLines = features.getTotalBlankLines() / (features.getTotalBlankLines() + entryNumLinesOfCode);
                double avgNumOfLoops = features.getNumOfLoops() / entryNumLinesOfCode;
                double avgNumOfAssignmentExpressions = features.getAssignExprs() / entryNumLinesOfCode;
                double avgNumOfNumbers = features.getNumbers() / entryNumLinesOfCode;

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

                // Non-aggregated
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
                pw.append(Integer.toString(features.getStatements()));
                pw.append(",");

                // Averages
                pw.append(Double.toString(avgNumOfConditionals));    
                pw.append(",");
                pw.append(Double.toString(avgNumOfLoops));
                pw.append(",");
                pw.append(Double.toString(avgNumOfAssignmentExpressions));
                pw.append(",");
                pw.append(Double.toString(avgNumOfComparisons));
                pw.append(",");
                pw.append(Double.toString(avgNumOfComments));
                pw.append(",");
                pw.append(Double.toString(avgNumOfArithmeticOperators));
                pw.append(",");
                pw.append(Double.toString(avgNumOfConditionals));
                pw.append(",");
                pw.append(Double.toString(avgCommas));
                pw.append(",");
                pw.append(Double.toString(avgParenthesis));
                pw.append(",");
                pw.append(Double.toString(avgPeriods));
                pw.append(",");
                pw.append(Double.toString(avgSpaces));
                pw.append(",");
                pw.append(Double.toString(avgLineLength));
                pw.append(",");
                pw.append(Float.toString(features.getMaxLineLength()));
                pw.append(",");
                pw.append(Double.toString(avgIndentationLength));
                pw.append(",");
                pw.append(Float.toString(features.getMaxIndentation()));
                pw.append(",");
                pw.append(Double.toString(avgBlankLines));
                pw.append(",");
                pw.append(Double.toString(avgNumOfNumbers));
                pw.append(",");

                // Maximums
                pw.append(Integer.toString(features.findMaxNumbers()));

                // Visual features
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getKeywordsX()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getIdentifiersX()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getOperatorsX()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getNumbersX()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getStringsX()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getLiteralsX()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getCommentsX()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getKeywordsY()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getIdentifiersY()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getOperatorsY()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getNumbersY()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getStringsY()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getLiteralsY()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getCommentsY()));
                pw.append("\n");
              })
          .explore(projectDir);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  /**
   * Seaches through loc_data.csv (stored as a List of String arrays) to find the entry for the file 
   * currently being parsed. The path also has to be modified as it is written with "\" in the 
   * DirExplorer and with "/" in loc_data.csv.
   */
  private static int findCorrespondingEntry(List<String[]> lines, String fileName) {
    int index = -1;
    int ctr = 0;
    fileName = fileName.replace("\\", "/");
    for (String[] line : lines) {
      if (line[1].endsWith(fileName)) {
        index = ctr;
        break;
      }
      ctr++;
    }
    return index;
  }
}
