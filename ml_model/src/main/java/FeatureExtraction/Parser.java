package FeatureExtraction;

import java.io.File;
import java.io.FileNotFoundException;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

import FeatureExtraction.FeatureExtractor.FeatureVisitor;

public class Parser {
    public static void main(String[] args) {
        
        FeatureExtractor featureExtractor = new FeatureExtractor();
        
        String dirPath = args[1];
        File projectDir = new File(dirPath);
        
        new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
            FeatureVisitor featureVisitor = featureExtractor.new FeatureVisitor();
            CompilationUnit cu = null;
            try {
                cu = StaticJavaParser.parse(file);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            featureVisitor.visit(cu,null);

            // TODO: Add the extracted features to the CSV file

        }).explore(projectDir); 
    }
}
