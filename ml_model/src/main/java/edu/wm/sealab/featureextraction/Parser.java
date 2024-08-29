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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Parser {
  public static void main(String[] args) {

    String dirPath = "/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/main/resources/corrected_raw_snippets/ds_6";
    File projectDir = new File(dirPath);

    // Output features
    File csvOutputFile = new File("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/feature_data.csv");
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

      // Non-aggregated features
      pw.append("#parameters");
      pw.append(",");
      // pw.append("#if statements");
      // pw.append(",");
      // pw.append("#loops");
      // pw.append(",");
      pw.append("#assignments");
      pw.append(",");
      pw.append("#commas");
      pw.append(",");
      pw.append("#periods");
      pw.append(",");
      pw.append("#spaces");
      pw.append(",");
      // pw.append("#comparisons");
      // pw.append(",");
      // pw.append("#parenthesis");
      // pw.append(",");
      pw.append("#literals");
      pw.append(",");
      pw.append("#statements");
      pw.append(",");
      pw.append("Volume");
      pw.append(",");
      pw.append("Entropy");
      pw.append(",");
      pw.append("#identifiers");
      pw.append(",");

      // Averages
      pw.append("#conditionals (avg)");
      pw.append(",");
      pw.append("#loops (avg)");
      pw.append(",");
      pw.append("#assignments (avg)");
      pw.append(",");
      pw.append("#comparisons (avg)");
      pw.append(",");
      pw.append("#comments (avg)");
      pw.append(",");
      pw.append("#operators (avg)");
      pw.append(",");
      pw.append("Identifiers Length (avg)");
      pw.append(",");
      pw.append("#keywords (avg)");
      pw.append(",");
      pw.append("#commas (avg)");
      pw.append(",");
      pw.append("#parenthesis (avg)");
      pw.append(",");
      pw.append("#periods (avg)");
      pw.append(",");
      pw.append("#spaces (avg)");
      pw.append(",");
      pw.append("Line length (avg)");
      pw.append(",");
      pw.append("Indentation length (avg)");
      pw.append(",");
      pw.append("#blank lines (avg)");
      pw.append(",");
      pw.append("#numbers (avg)");
      pw.append(",");
      pw.append("#identifiers (avg)");
      pw.append(",");
      pw.append("#nested blocks (avg)");
      pw.append(",");
      pw.append("MIDQ (avg)");
      pw.append(",");
      // pw.append("AEDQ (avg)");
      // pw.append(",");

      // Maximums
      pw.append("#numbers (max)");
      pw.append(",");
      pw.append("Identifiers length (max)");
      pw.append(",");
      pw.append("Indentation length (max)");
      pw.append(",");
      pw.append("Line length (max)");
      pw.append(",");
      pw.append("#keywords (max)");
      pw.append(",");
      pw.append("#identifiers (max)");
      pw.append(",");
      pw.append("#words (max)");
      pw.append(",");
      pw.append("MIDQ (max)");
      pw.append(",");
      // pw.append("AEDQ (max)");
      // pw.append(",");

      // Minimums
      pw.append("#identifiers (min)");
      pw.append(",");
      pw.append("MIDQ (min)");
      pw.append(",");
      // pw.append("AEDQ (min)");
      // pw.append(",");

      // dfts
      pw.append("#assignments (dft)");
      pw.append(",");
      pw.append("#commas (dft)");
      pw.append(",");
      pw.append("#comparisons (dft)");
      pw.append(",");
      pw.append("#comments (dft)");
      pw.append(",");
      pw.append("#conditionals (dft)");
      pw.append(",");
      pw.append("#identifiers (dft)");
      pw.append(",");
      pw.append("#keywords (dft)");
      pw.append(",");
      pw.append("#loops (dft)");
      pw.append(",");
      pw.append("#numbers (dft)");
      pw.append(",");
      pw.append("#operators (dft)");
      pw.append(",");
      pw.append("#parenthesis (dft)");
      pw.append(",");
      pw.append("#periods (dft)");
      pw.append(",");
      pw.append("#spaces (dft)");
      pw.append(",");
      pw.append("Indentation length (dft)");
      pw.append(",");
      pw.append("Line length (dft)");
      pw.append(",");

      //Visual Features
      pw.append("Keywords (Visual X)");
      pw.append(",");
      pw.append("Identifiers (Visual X)");
      pw.append(",");
      pw.append("Operators (Visual X)");
      pw.append(",");
      pw.append("Numbers (Visual X)");
      pw.append(",");
      pw.append("Strings (Visual X)");
      pw.append(",");
      pw.append("Literals (Visual X)");
      pw.append(",");
      pw.append("Comments (Visual X)");
      pw.append(",");
      pw.append("Keywords (Visual Y)");
      pw.append(",");
      pw.append("Identifiers (Visual Y)");
      pw.append(",");
      pw.append("Operators (Visual Y)");
      pw.append(",");
      pw.append("Numbers (Visual Y)");
      pw.append(",");
      pw.append("Strings (Visual Y)");
      pw.append(",");
      pw.append("Literals (Visual Y)");
      pw.append(",");
      pw.append("Comments (Visual Y)");
      pw.append(",");
      pw.append("Keywords (area)");
      pw.append(",");
      pw.append("Identifiers (area)");
      pw.append(",");
      pw.append("Operators (area)");
      pw.append(",");
      pw.append("Numbers (area)");
      pw.append(",");
      pw.append("Strings (area)");
      pw.append(",");
      pw.append("Literals (area)");
      pw.append(",");
      pw.append("Comments (area)");

      pw.append("\n");

      List<String[]> lines = null;
      try (FileReader fileReader = new FileReader("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/raw_loc_data.csv");
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
                visualFeatures.calculateVisualFeatures();

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
                
                HalsteadMetrics halsteadMetrics = new HalsteadMetrics(features);

                // Locate and extract file data from loc_data.csv
                int entryIndex = findCorrespondingEntry(allLines, file.getName());
                String[] entryLine = allLines.get(entryIndex);
                double entryNumLinesOfCode = Double.parseDouble(entryLine[4]) - 3 ; //subtract 2 for the class declaration and package declaration and last } of the class

                allLines.remove(entryIndex);

                // Calculate averages based on data from loc_data.csv
                double avgNumOfComments = features.getNumOfComments() / entryNumLinesOfCode;
                double avgNumOfComparisons = features.getComparisons() / entryNumLinesOfCode;
                double avgNumOfArithmeticOperators =
                    features.getArithmeticOperators() / entryNumLinesOfCode;
                double avgNumOfConditionals = features.getConditionals() / entryNumLinesOfCode;
                
                // Calculate average identifier length
                HashMap<String, List<String>> indetifierMap = features.getLineNumber_Identifier_Map();
                double avgIdentifierLength = 0.0;
                double totalIdentifierLength = 0.0;

                int totalNumIdentifiers = features.getIdentifiers();
                int maxNumIdentifiers = 0;
                ArrayList<Integer> numOfIdentifierPerLineList = new ArrayList<Integer>();

                for (String lineNumber : indetifierMap.keySet()) {
                  List<String> identifierList = indetifierMap.get(lineNumber);
                  for (String identifier : identifierList) {
                    totalIdentifierLength += identifier.length(); // total length of identifiers
                    numOfIdentifierPerLineList.add(identifierList.size()); // keep track of total number of identifiers per line
                  }
                }

                // max number of identifiers in a single line
                for (int numIdentifiers : numOfIdentifierPerLineList) {
                  if (numIdentifiers > maxNumIdentifiers) {
                    maxNumIdentifiers = numIdentifiers;
                  }
                }
                // minimum number of identifiers in a single line
                int minNumIdentifiers = maxNumIdentifiers;
                for (int numIdentifiers : numOfIdentifierPerLineList) {
                  if (numIdentifiers < minNumIdentifiers) {
                    minNumIdentifiers = numIdentifiers;
                  }
                }
                // average number of identifiers per line
                double avgNumIdentifiers = totalNumIdentifiers / entryNumLinesOfCode;

                // average number of nested blocks
                double avgNumOfNestedBlocks = features.getNestedBlocks() / entryNumLinesOfCode;

                avgIdentifierLength = totalIdentifierLength / features.getIdentifiers();
                
                // write the identifiers to files
                String dirPathForSnippetIdentifiers = "complexity-verification-project/ml_model/src/main/resources/snippet_identifier_splitter_out/DS_1/";
                File dirForSnippetIdentifiers = new File(dirPathForSnippetIdentifiers);
                if (!dirForSnippetIdentifiers.exists()) {
                  dirForSnippetIdentifiers.mkdir();
                }
                // write all identifiers to files. file name is the same as the snippet file name
                String snippetFileName = file.getName().split("\\.")[0];
                String fileName = dirPathForSnippetIdentifiers + snippetFileName + ".java";

                try (PrintWriter pw2 = new PrintWriter(fileName)) {
                  for (String lineNumber : indetifierMap.keySet()) {
                    List<String> identifierList = indetifierMap.get(lineNumber);
                    for (String identifier : identifierList) {
                      pw2.print(identifier + " ");
                    }
                    pw2.println();
                  }
                } catch (FileNotFoundException e) {
                  e.printStackTrace();
                }
                
                int numberOfKeywords = 0;
                int maxNumberOfKeywords = 0;
                Map<String, List<String>> keywords = features.getKeywords();
                for (String lineNumber : keywords.keySet()) {
                  List<String> keywordList = keywords.get(lineNumber);
                  if (keywordList.size() > maxNumberOfKeywords) {
                    maxNumberOfKeywords = keywordList.size();
                  }
                  numberOfKeywords += keywordList.size();
                }

                double avgNumOfKeywords = numberOfKeywords / entryNumLinesOfCode;
                double avgCommas = features.getCommas() / entryNumLinesOfCode;
                double avgParenthesis = features.getParenthesis() / entryNumLinesOfCode;
                double avgPeriods = features.getPeriods() / entryNumLinesOfCode;
                double avgSpaces = features.getSpaces() / entryNumLinesOfCode;
                
                double avgIndentationLength = features.getTotalIndentation() / entryNumLinesOfCode;
                double avgBlankLines = features.getTotalBlankLines() / entryNumLinesOfCode;
                double avgNumOfLoops = features.getNumOfLoops() / entryNumLinesOfCode;
                double avgNumOfAssignmentExpressions = features.getAssignExprs() / entryNumLinesOfCode;
                double avgNumOfNumbers = features.getNumbers() / entryNumLinesOfCode;


                // DFT Features
                // bandwidth of the number of assignment expressions
                double [] numOfAssignmentArray = new double[cu.getEnd().get().line];
                
                for (String lineNumber : features.getLineAssignmentExpressionMap().keySet()) {
                  double numOfAssignments = features.getLineAssignmentExpressionMap().get(lineNumber);
                  numOfAssignmentArray[Integer.parseInt(lineNumber)] = numOfAssignments;
                  
                }
                long dft_assignment = Dorn_DFT_Feature.getDFTBandwith(numOfAssignmentArray);

                // bandwidth of the number of commas
                double [] numOfCommasArray = new double[methodBody.split("\n").length];
                for (String lineNumber : features.getLineCommaMap().keySet()) {
                  double numOfCommas = features.getLineCommaMap().get(lineNumber);
                  numOfCommasArray[Integer.parseInt(lineNumber)] = numOfCommas;
                }
                long dft_commas = Dorn_DFT_Feature.getDFTBandwith(numOfCommasArray);

                // bandwidth of the number of comparisons
                double [] numOfComparisonsArray = new double[cu.getEnd().get().line];
                for (String lineNumber : features.getLineComparisonMap().keySet()) {
                  double numOfComparisons = features.getLineComparisonMap().get(lineNumber);
                  numOfComparisonsArray[Integer.parseInt(lineNumber)] = numOfComparisons;
                }
                long dft_comparisons = Dorn_DFT_Feature.getDFTBandwith(numOfComparisonsArray);

                // bandwidth of the number of comments
                double [] numOfCommentsArray = new double[cu.getEnd().get().line];
                for (String lineNumber : features.getLineCommentMap().keySet()) {
                  double numOfComments = features.getLineCommentMap().get(lineNumber);
                  numOfCommentsArray[Integer.parseInt(lineNumber)] = numOfComments;
                }
                long dft_comments = Dorn_DFT_Feature.getDFTBandwith(numOfCommentsArray);

                // bandwidth of the number of conditionals
                double [] numOfConditionalsArray = new double[cu.getEnd().get().line];
                for (String lineNumber : features.getLineConditionalMap().keySet()) {
                  double numOfConditionals = features.getLineConditionalMap().get(lineNumber);
                  numOfConditionalsArray[Integer.parseInt(lineNumber)] = numOfConditionals;
                }
                long dft_conditionals = Dorn_DFT_Feature.getDFTBandwith(numOfConditionalsArray);

                // bandwidth of the number of identifiers
                double [] numOfIdentifiersArray = new double[cu.getEnd().get().line];
                for (String lineNumber : features.getLineNumber_Identifier_Map().keySet()) {
                  double numOfIdentifiers = features.getLineNumber_Identifier_Map().get(lineNumber).size();
                  numOfIdentifiersArray[Integer.parseInt(lineNumber)] = numOfIdentifiers;
                }
                long dft_identifiers = Dorn_DFT_Feature.getDFTBandwith(numOfIdentifiersArray);

                // bandwidth of the number of keywords
                double [] numOfKeywordsArray = new double[methodBody.split("\n").length];
                for (String lineNumber : features.getKeywords().keySet()) {
                  double numOfKeywords = features.getKeywords().get(lineNumber).size();
                  numOfKeywordsArray[Integer.parseInt(lineNumber)] = numOfKeywords;
                }
                long dft_keywords = Dorn_DFT_Feature.getDFTBandwith(numOfKeywordsArray);

                // bandwidth of the number of loops
                double [] numOfLoopsArray = new double[cu.getEnd().get().line];
                for (String lineNumber : features.getLineLoopMap().keySet()) {
                  double numOfLoops = features.getLineLoopMap().get(lineNumber);
                  numOfLoopsArray[Integer.parseInt(lineNumber)] = numOfLoops;
                }
                long dft_loops = Dorn_DFT_Feature.getDFTBandwith(numOfLoopsArray);

                // bandwidth of the number of numbers
                double [] numOfNumbersArray = new double[cu.getEnd().get().line];
                for (String lineNumber : features.getLineNumberMap().keySet()) {
                  double numOfNumbers = features.getLineNumberMap().get(lineNumber).size();
                  numOfNumbersArray[Integer.parseInt(lineNumber)] = numOfNumbers;
                }
                long dft_numbers = Dorn_DFT_Feature.getDFTBandwith(numOfNumbersArray);
                
                // bandwidth of the number of operators
                double [] numOfOperatorsArray = new double[cu.getEnd().get().line];
                for (String lineNumber : features.getLineOperatorMap().keySet()) {
                  double numOfOperators = features.getLineOperatorMap().get(lineNumber);
                  numOfOperatorsArray[Integer.parseInt(lineNumber)] = numOfOperators;
                }
                long dft_operators = Dorn_DFT_Feature.getDFTBandwith(numOfOperatorsArray);

                // bandwidth of the number of parenthesis
                double [] numOfParenthesisArray = new double[methodBody.split("\n").length];
                for (String lineNumber : features.getLineParenthesisMap().keySet()) {
                  double numOfParenthesis = features.getLineParenthesisMap().get(lineNumber);
                  numOfParenthesisArray[Integer.parseInt(lineNumber)] = numOfParenthesis;
                }
                long dft_parenthesis = Dorn_DFT_Feature.getDFTBandwith(numOfParenthesisArray);

                // bandwidth of the number of periods
                double [] numOfPeriodsArray = new double[methodBody.split("\n").length];
                for (String lineNumber : features.getLinePeriodMap().keySet()) {
                  double numOfPeriods = features.getLinePeriodMap().get(lineNumber);
                  numOfPeriodsArray[Integer.parseInt(lineNumber)] = numOfPeriods;
                }
                long dft_periods = Dorn_DFT_Feature.getDFTBandwith(numOfPeriodsArray);

                // bandwidth of the number of spaces
                double [] numOfSpacesArray = new double[methodBody.split("\n").length];
                for (String lineNumber : features.getLineSpaceMap().keySet()) {
                  double numOfSpaces = features.getLineSpaceMap().get(lineNumber);
                  numOfSpacesArray[Integer.parseInt(lineNumber)] = numOfSpaces;
                }
                long dft_spaces = Dorn_DFT_Feature.getDFTBandwith(numOfSpacesArray);

                // bandwidth of the number of indentation length
                double [] numOfIndentationLengthArray = new double[methodBody.split("\n").length];
                for (String lineNumber : features.getLineIndentationLengthMap().keySet()) {
                  double numOfIndentationLength = features.getLineIndentationLengthMap().get(lineNumber);
                  numOfIndentationLengthArray[Integer.parseInt(lineNumber)] = numOfIndentationLength;
                }
                long dft_indentationLength = Dorn_DFT_Feature.getDFTBandwith(numOfIndentationLengthArray);


                /* Need to consider the strings and characters in the method body */
                String methodBody_with_strings_charaters = cu.findFirst(MethodDeclaration.class)
                            .map(method -> method.toString())
                            .orElse("");
                Features features_with_strings_chracters = sfe.extract(methodBody);
                double avgLineLength = features_with_strings_chracters.getTotalLineLength() / (features_with_strings_chracters.getTotalBlankLines() + entryNumLinesOfCode);

                // bandwidth of the number of line length
                double [] numOfLineLengthArray = new double[methodBody_with_strings_charaters.split("\n").length];
                for (String lineNumber : features_with_strings_chracters.getLineLineLengthMap().keySet()) {
                  double numOfLineLength = features_with_strings_chracters.getLineLineLengthMap().get(lineNumber);
                  numOfLineLengthArray[Integer.parseInt(lineNumber)] = numOfLineLength;
                }
                long dft_lineLength = Dorn_DFT_Feature.getDFTBandwith(numOfLineLengthArray);

                // maximum number of words in a single line
                int maxNumOfWords = features_with_strings_chracters.getMaxWords();

                // Calculate the min, max and avg of MIDQ

                DocumentRelatedFeatures drf = new DocumentRelatedFeatures(features);
                
                double sumMIDQ = 0;
                double maxMIDQ = 0;
                double minMIDQ = 100;
                
                List<Double> MIDQList = drf.getMIDQ();
                for (double midq : MIDQList) {
                  sumMIDQ += midq;
                  if (midq > maxMIDQ) {
                    maxMIDQ = midq;
                  }
                  if (midq < minMIDQ) {
                    minMIDQ = midq;
                  }
                }
                double avgMIDQ = sumMIDQ / entryNumLinesOfCode;



                // TODO: Need to implement this logic //
                // Calculate the min, max and avg of AEDQ
                // double sumAEDQ = 0;
                // double maxAEDQ = 0;
                // double minAEDQ = 100;

                // /////////// TEMPOARY COMMENTED OUT //////////////
                // List<Integer> AEDQList = drf.getAEDQ();
                // // List<Integer> AEDQList = new ArrayList<Integer>();
                // for (double aedq : AEDQList) {
                //   sumAEDQ += aedq;
                //   if (aedq > maxAEDQ) {
                //     maxAEDQ = aedq;
                //   }
                //   if (aedq < minAEDQ) {
                //     minAEDQ = aedq;
                //   }
                // }
                // double avgAEDQ = sumAEDQ / AEDQList.size(); // because AEDQ is not per line but per class/API call



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
                // pw.append(Integer.toString(features.getNumOfIfStatements()));
                // pw.append(",");
                // pw.append(Integer.toString(features.getNumOfLoops()));
                // pw.append(",");
                pw.append(Integer.toString(features.getAssignExprs()));
                pw.append(",");
                pw.append(Integer.toString(features.getCommas()));
                pw.append(",");
                pw.append(Integer.toString(features.getPeriods()));
                pw.append(",");
                pw.append(Integer.toString(features.getSpaces()));
                pw.append(",");
                // pw.append(Integer.toString(features.getComparisons()));
                // pw.append(",");
                // pw.append(Integer.toString(features.getParenthesis()));
                // pw.append(",");
                pw.append(Integer.toString(features.getLiterals()));
                pw.append(",");
                pw.append(Integer.toString(features.getStatements()));
                pw.append(",");
                pw.append(Double.toString(halsteadMetrics.getProgramVolume()));
                pw.append(",");
                pw.append(Double.toString(features.getTokenEntropy()));
                pw.append(",");
                pw.append(Integer.toString(features.getIdentifiers()));
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
                pw.append(Double.toString(avgIdentifierLength));
                pw.append(",");
                pw.append(Double.toString(avgNumOfKeywords));
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
                pw.append(Double.toString(avgIndentationLength));
                pw.append(",");
                pw.append(Double.toString(avgBlankLines));
                pw.append(",");
                pw.append(Double.toString(avgNumOfNumbers));
                pw.append(",");
                pw.append(Double.toString(avgNumIdentifiers));
                pw.append(",");
                pw.append(Double.toString(avgNumOfNestedBlocks));
                pw.append(",");
                pw.append(Double.toString(avgMIDQ));
                pw.append(",");
                // pw.append(Double.toString(avgAEDQ));
                // pw.append(",");

                // Maximums
                pw.append(Integer.toString(features.findMaxNumbers()));
                pw.append(",");
                pw.append(Integer.toString(features.getMaxIdentifierLength()));
                pw.append(",");
                pw.append(Float.toString(features.getMaxIndentation()));
                pw.append(",");
                pw.append(Float.toString(features.getMaxLineLength()));
                pw.append(",");
                pw.append(Integer.toString(maxNumberOfKeywords));
                pw.append(",");
                pw.append(Integer.toString(maxNumIdentifiers));
                pw.append(",");
                pw.append(Integer.toString(maxNumOfWords));
                pw.append(",");
                pw.append(Double.toString(maxMIDQ));
                pw.append(",");
                // pw.append(Double.toString(maxAEDQ));
                // pw.append(",");

                // Minimums
                pw.append(Integer.toString(minNumIdentifiers));
                pw.append(",");
                pw.append(Double.toString(minMIDQ));
                pw.append(",");
                // pw.append(Double.toString(minAEDQ));
                // pw.append(",");

                // dfts
                pw.append(Long.toString(dft_assignment));
                pw.append(",");
                pw.append(Long.toString(dft_commas));
                pw.append(",");
                pw.append(Long.toString(dft_comparisons));
                pw.append(",");
                pw.append(Long.toString(dft_comments));
                pw.append(",");
                pw.append(Long.toString(dft_conditionals));
                pw.append(",");
                pw.append(Long.toString(dft_identifiers));
                pw.append(",");
                pw.append(Long.toString(dft_keywords));
                pw.append(",");
                pw.append(Long.toString(dft_loops));
                pw.append(",");
                pw.append(Long.toString(dft_numbers));
                pw.append(",");
                pw.append(Long.toString(dft_operators));
                pw.append(",");
                pw.append(Long.toString(dft_parenthesis));
                pw.append(",");
                pw.append(Long.toString(dft_periods));
                pw.append(",");
                pw.append(Long.toString(dft_spaces));
                pw.append(",");
                pw.append(Long.toString(dft_indentationLength));
                pw.append(",");
                pw.append(Long.toString(dft_lineLength));
                
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
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getKeywordsArea()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getIdentifiersArea()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getOperatorsArea()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getNumbersArea()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getStringsArea()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getLiteralsArea()));
                pw.append(",");
                pw.append(Double.toString(visualFeatures.getCommentsArea()));

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
      if (line[2].endsWith(fileName)) {
        index = ctr;
        break;
      }
      ctr++;
    }
    return index;
  }
}
