package edu.wm.sealab.featureextraction;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;


/**
 * This class to extract features from a java snippet
 * Input : Java file with a Single Method (Note: if want to use all the features)
 * Output: Features of the java file
 * */ 
// public class FeatureExtractor {

    public class FeatureVisitor extends VoidVisitorAdapter<Void> {

        // features of a java snippet
        private Features features = new Features();

        /**
         * This method to compute #parameters of a java method
         */
        @Override
        public void visit(MethodDeclaration md, Void arg){
            super.visit(md, arg);
            if (md.getBody().isPresent()) {
                // set #parameters of method
                features.setNumOfParameters(md.getParameters().size());
            }
        }

        /**
         * This method to compute #if statements of a java method
         */
        @Override
        public void visit(IfStmt ifStmt, Void arg) {
            super.visit(ifStmt, arg);
            features.setNumOfIfStatements(features.getNumOfIfStatements() + 1);
        }

        /**
         * This method to compute #for loops of a java method
         */
        @Override
        public void visit(ForStmt forStmt, Void arg) {
            super.visit(forStmt, arg);
            features.setNumOfLoops(features.getNumOfLoops() + 1);
        }

        /**
         * This method to compute # while loops of a java method
         */
        @Override
        public void visit(WhileStmt whileStmt, Void arg) {
            super.visit(whileStmt, arg);
            features.setNumOfLoops(features.getNumOfLoops() + 1);
        }

        /**
         * This method to get the computed features
         */
        public Features getFeatures() {
            return features;
        }
    }

// }
