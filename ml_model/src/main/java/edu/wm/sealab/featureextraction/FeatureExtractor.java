package edu.wm.sealab.featureextraction;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;


/**
 * This class to extract features from a java file
 * Input : Java file with a Single Method (Note: if want to use all the features)
 * Output: FeatureMap
 * */ 
public class FeatureExtractor {

    public class FeatureVisitor extends VoidVisitorAdapter<Void> {

        private FeatureMap featureMap = new FeatureMap();

        /**
         * This method to compute #parameters of a java method
         * @param MethodDeclaration
         * @param Void
         */
        @Override
        public void visit(MethodDeclaration md, Void arg){
            super.visit(md, arg);
            if (md.getBody().isPresent()) {
                // set #parameters of method
                featureMap.setNumOfParameters(md.getParameters().size());
            }
        }

        /**
         * This method to compute #if statements of a java method
         * @param IfStmt
         * @param Void
         */
        @Override
        public void visit(IfStmt ifStmt, Void arg) {
            super.visit(ifStmt, arg);
            featureMap.setNumOfIfStatements(featureMap.getNumOfIfStatements() + 1);
        }

        /**
         * This method to compute # for loops of a java method
         * @param ForStmt
         * @param Void
         */
        @Override
        public void visit(ForStmt forStmt, Void arg) {
            super.visit(forStmt, arg);
            featureMap.setNumOfLoops(featureMap.getNumOfLoops() + 1);
        }

        /**
         * This method to compute # while loops of a java method
         * @param WhileStmt
         * @param Void
         */
        @Override
        public void visit(WhileStmt whileStmt, Void arg) {
            super.visit(whileStmt, arg);
            featureMap.setNumOfLoops(featureMap.getNumOfLoops() + 1);
        }

        /**
         * This method to get the computed features
         * @return FeatureMap
         */
        public FeatureMap getFeatures() {
            return featureMap;
        }
    }

}
