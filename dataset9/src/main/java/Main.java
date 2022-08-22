public class Main {
    public static void main(String[] args) {
        System.out.print(System.getProperty("java.class.path"));
        CodeSnippets cs = new CodeSnippets();
        cs.runAll();
    }
}
