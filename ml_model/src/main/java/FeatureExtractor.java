import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class FeatureExtractor {

    private static Map<String, Object> features = null;

    public static class FeatureVisitor extends VoidVisitorAdapter<Void> {

        private static final String NUM_OF_PARAMETERS = "#parameters";
        private static final String IF_STATEMENTS = "ifStmts";
        private static final String LOOPS = "loops";

        // keep all the features
        Map<String,Object> features = new HashMap<>();

        // feature 1: #parameters
        private  int numOfParameters = 0;

        // feature 2: #if statements
        Set<String> ifStatements = new HashSet<>();

        // feature 3: #loops
        Set<String> loops = new HashSet<>();

        @Override
        public void visit(MethodDeclaration md, Void arg){
            super.visit(md, arg);
            if (md.getBody().isPresent()) {

                // get #parameters of method
                numOfParameters = md.getParameters().size();
            }
        }

        @Override
        public void visit(IfStmt ifStmt, Void arg) {
            super.visit(ifStmt, arg);
            ifStatements.add(ifStmt.toString());
        }

        @Override
        public void visit(ForStmt forStmt, Void arg) {
            super.visit(forStmt, arg);
            loops.add(forStmt.toString());
        }

        @Override
        public void visit(WhileStmt whileStmt, Void arg) {
            super.visit(whileStmt, arg);
            loops.add(whileStmt.toString());
        }

        public Map<String,Object> getFeatures() {
            features.put(NUM_OF_PARAMETERS, numOfParameters);
            features.put(IF_STATEMENTS, ifStatements);
            features.put(LOOPS, loops);
            return features;
        }
    }

    public static void main(String[] args) {
        FeatureVisitor featureVisitor = new FeatureVisitor();

        String FILE_PATH = "simple-datasets/src/main/java/cog_complexity_validation_datasets/One";
        File projectDir = new File(FILE_PATH);


        new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
            CompilationUnit cu = null;
            try {
                cu = StaticJavaParser.parse(file);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            featureVisitor.visit(cu,null);

            // Uncomment If need to get the  individual features list
            // features = featureVisitor.getFeatures();

            System.out.println(file.getName());
            System.out.println("Number of if statements: "+featureVisitor.ifStatements.size());
            System.out.println("Number of loops: "+featureVisitor.loops.size());

        }).explore(projectDir);

        
    }

}
