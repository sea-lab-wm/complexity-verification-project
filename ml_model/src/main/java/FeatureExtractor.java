import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.Statement;

import com.google.common.base.Strings;
import java.io.File;
import java.io.IOException;

public class FeatureExtractor {


    // List all classes in a project
    public static void listClasses(File projectDir) {
        new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
            System.out.println(path);
            System.out.println(Strings.repeat("=", path.length()));
            try {
                new VoidVisitorAdapter<Object>() {
                    @Override
                    public void visit(ClassOrInterfaceDeclaration n, Object arg) {
                        super.visit(n, arg);
                        System.out.println(" * " + n.getName());
                    }
                }.visit(StaticJavaParser.parse(file), null);
                System.out.println(); // empty line
            } catch (IOException e) {
                new RuntimeException(e);
            }
        }).explore(projectDir);
    }

    // List all methods in a project
    public static void listMethods(File projectDir) {
        new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
            System.out.println(path);
            System.out.println(Strings.repeat("=", path.length()));
            try {
                new VoidVisitorAdapter<Object>() {
                    @Override
                    public void visit(ClassOrInterfaceDeclaration n, Object arg) {
                        super.visit(n, arg);
                        n.getMethods().forEach(m -> System.out.println(" * " + m.getDeclarationAsString()));
                    }
                }.visit(StaticJavaParser.parse(file), null);
                System.out.println(); // empty line
            } catch (IOException e) {
                new RuntimeException(e);
            }
        }).explore(projectDir);
    }

    // list all if statements in a project
    public static void listIfStatements(File projectDir) {
        new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
            System.out.println(path);
            System.out.println(Strings.repeat("=", path.length()));
            try {
                new VoidVisitorAdapter<Object>() {
                    @Override
                    public void visit(ClassOrInterfaceDeclaration n, Object arg) {
                        super.visit(n, arg);
                        n.getMethods().forEach(m -> m.getBody().ifPresent(b -> b.getStatements().forEach(s -> {
                            if (s.isIfStmt()) {
                                System.out.println(" * " + s);
                            }
                        })));
                    }
                }.visit(StaticJavaParser.parse(file), null);
                System.out.println(); // empty line
            } catch (IOException e) {
                new RuntimeException(e);
            }
        }).explore(projectDir);
    }
    
    // list all parameters in a project
    public static void listParameters(File projectDir) {
        new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
            System.out.println(path);
            System.out.println(Strings.repeat("=", path.length()));
            try {
                new VoidVisitorAdapter<Object>() {
                    @Override
                    public void visit(ClassOrInterfaceDeclaration n, Object arg) {
                        super.visit(n, arg);
                        n.getMethods().forEach(m -> m.getParameters().forEach(s -> {
                            System.out.println(" * " + s);
                        }));
                    }
                }.visit(StaticJavaParser.parse(file), null);
                System.out.println(); // empty line
            } catch (IOException e) {
                new RuntimeException(e);
            }
        }).explore(projectDir);    
    }

    // list all loops in a project
    public static void listLoopStatements(File projectDir) {
        new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
            System.out.println(path);
            System.out.println(Strings.repeat("=", path.length()));
            try {
                new VoidVisitorAdapter<Object>() {
                    @Override
                    public void visit(ClassOrInterfaceDeclaration n, Object arg) {
                        super.visit(n, arg);
                        
                        n.getMethods().forEach(m -> m.getBody().ifPresent(b -> b.getStatements().forEach(s -> {
                            getAllNestedLoops(s);
                        })));
                    }

                    private void getAllNestedLoops(Statement s) {
                        if (s.isForStmt()) {
                            System.out.println(" * " + s);
                            getAllNestedLoops(s.asForStmt().getBody());
                        }

                        if (s.isForEachStmt()) {
                            System.out.println(" * " + s);
                            getAllNestedLoops(s.asForEachStmt().getBody());
                        }

                        if (s.isWhileStmt()) {
                            System.out.println(" * " + s);
                            getAllNestedLoops(s.asWhileStmt().getBody());
                        }

                        System.out.println(" ??? " + s);
                    }
                }.visit(StaticJavaParser.parse(file), null);
                System.out.println(); // empty line
            } catch (IOException e) {
                new RuntimeException(e);
            }
        }).explore(projectDir);
    }

}
