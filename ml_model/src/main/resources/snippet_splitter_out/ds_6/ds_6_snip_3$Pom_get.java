package snippet_splitter_out.ds_6;

public class ds_6_snip_3$Pom_get {
  // hudson.os.PosixAPI.get()
  // SNIPPET_STARTS
  @Deprecated
  public static synchronized org.jruby.ext.posix.POSIX get() {
    if (jnaPosix == null) {
      jnaPosix =
          org.jruby.ext.posix.POSIXFactory.getPOSIX(
              new // Change POSIXHandler signature
              Pom.POSIXHandler() {

                public void error(ERRORS errors, String s) throws PosixException {
                  // changed ERRORS signature, added throws
                  throw new PosixException(s, errors);
                }

                public void unimplementedError(String s) {
                  throw new UnsupportedOperationException(s);
                }

                public void warn(WARNING_ID warning_id, String s, Object... objects) {
                  LOGGER.fine(s);
                }

                public boolean isVerbose() {
                  return true;
                }

                public File getCurrentWorkingDirectory() {
                  return new File(".").getAbsoluteFile();
                }

                public String[] getEnv() {
                  Map<String, String> envs = System.getenv();
                  String[] envp = new String[envs.size()];
                  int i = 0;
                  for (Map.Entry<String, String> e : envs.entrySet()) {
                    envp[i++] = e.getKey() + '+' + e.getValue();
                  }
                  return envp;
                }

                public InputStream getInputStream() {
                  return System.in;
                }

                public PrintStream getOutputStream() {
                  return System.out;
                }

                public int getPID() {
                  // TODO
                  return 0;
                }

                public PrintStream getErrorStream() {
                  return System.err;
                }
              },
              true);
    }
    return jnaPosix;
  }
}
