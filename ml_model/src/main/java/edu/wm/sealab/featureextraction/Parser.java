package edu.wm.sealab.featureextraction;

import java.io.File;
import java.io.FileNotFoundException;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

// import edu.wm.sealab.featureextraction.FeatureExtractor.FeatureVisitor;

public class Parser {
    public static void main(String[] args) {
        
        FeatureVisitor featureVisitor = new FeatureVisitor();

        if (args.length != 1) {
            System.out.println("Usage: java Parser <project_dir>");
            System.exit(1);
        }
        String dirPath = args[0];
        File projectDir = new File(dirPath);

        new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
            CompilationUnit cu = null;
            try {
                cu = StaticJavaParser.parse(file);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            featureVisitor.visit(cu,null);
            System.out.println(featureVisitor.getFeatures().getNumOfIfStatements());

            // TODO: Add the extracted features to the CSV file
        }).explore(projectDir);
    }
}
