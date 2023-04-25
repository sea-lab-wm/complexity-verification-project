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

        
    }
}
