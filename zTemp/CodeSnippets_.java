import hirondelle.web4j.util.Util;
import hirondelle.web4j.webmaster.TroubleTicket;

import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.security.Principal;
import java.util.Collection;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.logging.Logger;

import javax.servlet.AsyncContext;
import javax.servlet.DispatcherType;
import javax.servlet.RequestDispatcher;
import javax.servlet.ServletContext;
import javax.servlet.ServletException;
import javax.servlet.ServletInputStream;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import javax.servlet.http.HttpUpgradeHandler;
import javax.servlet.http.Part;

public class CodeSnippets {

    //ADDED BY KOBI
    public void runAll() {
        HttpServletRequest h = new HttpServletRequest() {
            @Override
            public AsyncContext getAsyncContext() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Object getAttribute(String arg0) {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Enumeration<String> getAttributeNames() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getCharacterEncoding() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public int getContentLength() {
                // TODO Auto-generated method stub
                return 0;
            }
            @Override
            public long getContentLengthLong() {
                // TODO Auto-generated method stub
                return 0;
            }
            @Override
            public String getContentType() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public DispatcherType getDispatcherType() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public ServletInputStream getInputStream() throws IOException {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getLocalAddr() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getLocalName() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public int getLocalPort() {
                // TODO Auto-generated method stub
                return 0;
            }
            @Override
            public Locale getLocale() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Enumeration<Locale> getLocales() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getParameter(String arg0) {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Map<String, String[]> getParameterMap() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Enumeration<String> getParameterNames() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String[] getParameterValues(String arg0) {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getProtocol() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public BufferedReader getReader() throws IOException {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getRealPath(String arg0) {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getRemoteAddr() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getRemoteHost() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public int getRemotePort() {
                // TODO Auto-generated method stub
                return 0;
            }
            @Override
            public RequestDispatcher getRequestDispatcher(String arg0) {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getScheme() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getServerName() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public int getServerPort() {
                // TODO Auto-generated method stub
                return 0;
            }
            @Override
            public ServletContext getServletContext() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public boolean isAsyncStarted() {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public boolean isAsyncSupported() {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public boolean isSecure() {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public void removeAttribute(String arg0) {
                // TODO Auto-generated method stub
            }
            @Override
            public void setAttribute(String arg0, Object arg1) {
                // TODO Auto-generated method stub
            }
            @Override
            public void setCharacterEncoding(String arg0) throws UnsupportedEncodingException {
                // TODO Auto-generated method stub
            }
            @Override
            public AsyncContext startAsync() throws IllegalStateException {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public AsyncContext startAsync(ServletRequest arg0, ServletResponse arg1) throws IllegalStateException {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public boolean authenticate(HttpServletResponse arg0) throws IOException, ServletException {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public String changeSessionId() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getAuthType() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getContextPath() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Cookie[] getCookies() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public long getDateHeader(String arg0) {
                // TODO Auto-generated method stub
                return 0;
            }
            @Override
            public String getHeader(String arg0) {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Enumeration<String> getHeaderNames() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Enumeration<String> getHeaders(String arg0) {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public int getIntHeader(String arg0) {
                // TODO Auto-generated method stub
                return 0;
            }
            @Override
            public String getMethod() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Part getPart(String arg0) throws IOException, ServletException {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Collection<Part> getParts() throws IOException, ServletException {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getPathInfo() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getPathTranslated() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getQueryString() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getRemoteUser() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getRequestURI() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public StringBuffer getRequestURL() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getRequestedSessionId() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public String getServletPath() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public HttpSession getSession() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public HttpSession getSession(boolean arg0) {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public Principal getUserPrincipal() {
                // TODO Auto-generated method stub
                return null;
            }
            @Override
            public boolean isRequestedSessionIdFromCookie() {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public boolean isRequestedSessionIdFromURL() {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public boolean isRequestedSessionIdFromUrl() {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public boolean isRequestedSessionIdValid() {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public boolean isUserInRole(String arg0) {
                // TODO Auto-generated method stub
                return false;
            }
            @Override
            public void login(String arg0, String arg1) throws ServletException {
                // TODO Auto-generated method stub
            }
            @Override
            public void logout() throws ServletException {
                // TODO Auto-generated method stub
            }
            @Override
            public <T extends HttpUpgradeHandler> T upgrade(Class<T> arg0) throws IOException, ServletException {
                // TODO Auto-generated method stub
                return null;
            }
        };

        /*
        logAndEmailSeriousProblemS111(new Throwable(), h);
        logAndEmailSeriousProblemS112(new Throwable(), h);
        logAndEmailSeriousProblemS113(new Throwable(), h);
        logAndEmailSeriousProblemS121(new Throwable(), h);
        logAndEmailSeriousProblemS122(new Throwable(), h);
        logAndEmailSeriousProblemS123(new Throwable(), h);

        apply211(new Project(), new NotificationStore());
        apply212(new Project(), new NotificationStore());
        apply213(new Project(), new NotificationStore());
        apply(new Project(), new NotificationStore());
        apply222(new Project(), new NotificationStore());
        apply223(new Project(), new NotificationStore());

        copyUTable311(new UTable());
        copyUTable312(new UTable());
        copyUTable313(new UTable());
        writeUTable321(new UTable());
        writeUTable(new UTable());
        writeUTable323(new UTable());

        getMove411(new StringBuilder(), new Move());
        getMove412(new StringBuilder(), new Move());
        getMove413(new StringBuilder(), new Move());
        getMove421(new StringBuilder(), new Move());
        getMove422(new StringBuilder(), new Move());
        getMove(new StringBuilder(), new Move());

        shortenText511(new String(), new Control());
        shortenText512(new String(), new Control());
        shortenText513(new String(), new Control());
        shortenText521(new String(), new Control());
        shortenText522(new String(), new Control());
        shortenText523(new String(), new Control());
        */
    }


    // Snippet 1
    // hirondelle.web4j.Controller.logAndEmailSeriousProblem
    // http://www.web4j.com/web4j/javadoc/src‐html/hirondelle/web4j/Controller.html#line.381

    /**
     Key name for the most recent {@link TroubleTicket}, placed in application scope when a
     problem occurs.
     <P>Key name: {@value}.
     */
    public static final String MOST_RECENT_TROUBLE_TICKET = "web4j_key_for_most_recent_trouble_ticket";
    private static final Logger fLogger = Util.getLogger(CodeSnippets.class);
    private static final String ELLIPSIS = "";
    private static final String NOTIFICATION_COMPOSITE = "";

    // S1_1:1 method chains, good comments
    /**
    * Informs the webmaster of an unexpected problem (Exception "ex")
    * with the deployed application (indicated by “aRequest”).
    */
    //SNIPPET_STARTS
    public void logAndEmailSeriousProblemS111(Throwable ex, HttpServletRequest aRequest)
    {
    /* Create trouble ticket with context reference. */
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    /* Log message to file. */
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    /* Log message to output. */
    System.out.println("SERIOUS PROBLEM OCCURRED."); // changed to allow compilation
    System.out.println(troubleTicket.toString());// changed to allow compilation
    /* Remember most recent ticket and inform webmaster. */
    aRequest.getSession().getServletContext().
    setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    troubleTicket.toString();// changed to allow compilation
    }

    // S1_1:2 method chains, bad comments
    /**
    * Informs the webmaster of an unexpected problem (Exception “ex”)
    * with the deployed application (indicated by “aRequest”).
    */
    //SNIPPET_STARTS
    public void logAndEmailSeriousProblemS112(Throwable ex, HttpServletRequest aRequest)
    {
    /* Define local variable. */
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    /* Log message. */
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    /* Log message again. */
    System.out.println("SERIOUS PROBLEM OCCURRED.");// changed to allow compilation
    System.out.println(troubleTicket.toString());// changed to allow compilation
    /* Update context and mail trouble ticket. */
    aRequest.getSession().getServletContext().
    setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    }

    // S1_1:3 method chains, no comments
    /**
    * Informs the webmaster of an unexpected problem (Exception “ex”)
    * with the deployed application (indicated by “aRequest”).
    */
    //SNIPPET_STARTS
    public void logAndEmailSeriousProblemS113(Throwable ex, HttpServletRequest aRequest)
    {
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    System.out.println("SERIOUS PROBLEM OCCURRED.");// changed to allow compilation
    System.out.println(troubleTicket.toString());// changed to allow compilation
    aRequest.getSession().getServletContext().
    setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    troubleTicket.toString(); // changed to allow compilation
    }

    // S1_2:1 resolved method chains, good comments
    /**
    * Informs the webmaster of an unexpected problem (Exception “ex”)
    * with the deployed application (indicated by “aRequest”).
    */
    //SNIPPET_STARTS
    public void logAndEmailSeriousProblemS121(Throwable ex, HttpServletRequest aRequest)
    {
    /* Create trouble ticket with context reference. */
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    /* Log message to file. */
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    /* Log message to output. */
    System.out.println("SERIOUS PROBLEM OCCURRED."); // changed to allow compilation
    System.out.println(troubleTicket.toString()); // changed to allow compilation
    /* Remember most recent ticket and inform webmaster. */
    HttpSession session = aRequest.getSession();
    ServletContext context = session.getServletContext();
    context.setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    troubleTicket.toString(); // changed to allow compilation
    }

    // S1_2:2 resolved method chains, bad comments
    /**
    * Informs the webmaster of an unexpected problem (Exception “ex”)
    * with the deployed application (indicated by “aRequest”).
    */
    //SNIPPET_STARTS
    public void logAndEmailSeriousProblemS122(Throwable ex, HttpServletRequest aRequest)
    {
    /* Define local variable. */
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    /* Log message. */
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    /* Log message again. */
    System.out.println("SERIOUS PROBLEM OCCURRED.");// changed to allow compilation
    System.out.println(troubleTicket.toString());// changed to allow compilation
    /* Update context and mail trouble ticket. */
    HttpSession session = aRequest.getSession();
    ServletContext context = session.getServletContext();
    context.setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    troubleTicket.toString();// changed to allow compilation
    }

    // S1_2:3 resolved method chains, no comments
    /**
    * Informs the webmaster of an unexpected problem (Exception “ex”)
    * with the deployed application (indicated by “aRequest”).
    */
    //SNIPPET_STARTS
    public void logAndEmailSeriousProblemS123(Throwable ex, HttpServletRequest aRequest)
    {
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    System.out.println("SERIOUS PROBLEM OCCURRED.");// changed to allow compilation
    System.out.println(troubleTicket.toString());// changed to allow compilation
    HttpSession session = aRequest.getSession();
    ServletContext context = session.getServletContext();
    context.setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    troubleTicket.toString();// changed to allow compilation
    }



    

    

    

    

    //private class Project {
    //    public ProjectSpace eContainer() {
    //        return null;
    //    }
   // }

    //private class ProjectSpace {
    //    public PropertyManager getPropertyManager() {
    //        return null;
    //    }
    //}



    //private class PropertyManager {
    //    public StoreProperty getLocalProperty(String notificationComposite) {
    //        return null;
    //    }

    //    public void setLocalProperty(String notificationComposite, NotificationComposite nComposite) {

    //    }
    //}

    //private class StoreProperty {
    //    public Value getValue() {
    //        return null;
    //    }
    //}

    private class Value {
        public void test() {

        }
    }

    private class NotificationComposite {
        //public <E> NotificationList getNotifications() {
        //    return null;
        //}
    }

    private static class Factory {
        public static NotificationComposite createNotificationComposite() {
            return null;
        }
    }

    private class NotificationList implements Collection {
        @Override
        public int size() {
            return 0;
        }

        @Override
        public boolean isEmpty() {
            return false;
        }

        @Override
        public boolean contains(Object o) {
            return false;
        }

        @Override
        public Iterator iterator() {
            return null;
        }

        @Override
        public Object[] toArray() {
            return new Object[0];
        }

        @Override
        public boolean add(Object o) {
            return false;
        }

        @Override
        public boolean remove(Object o) {
            return false;
        }

        @Override
        public boolean addAll(Collection c) {
            return false;
        }

        @Override
        public void clear() {

        }

        @Override
        public boolean equals(Object o) {
            return false;
        }

        @Override
        public int hashCode() {
            return 0;
        }

        @Override
        public boolean retainAll(Collection c) {
            return false;
        }

        @Override
        public boolean removeAll(Collection c) {
            return false;
        }

        @Override
        public boolean containsAll(Collection c) {
            return false;
        }

        @Override
        public Object[] toArray(Object[] a) {
            return new Object[0];
        }


    }

    

    

    private class UImage {
        public Path getPath() {
            return null;
        }
    }

    private class UColor {
        public int getRed() {
            return 0;
        }

        public int getGreen() {
            return 0;
        }

        public int getBlue() {
            return 0;
        }
    }

    private class UChildren {
        public UImage get(int i) {
            return null;
        }

        public int size() {
            return 0;
        }
    }

    private class Path {
        public USegment lastSegment() {
            return null;
        }
    }

    private class USegment {
        public boolean startsWith(String aFalse) {
            return false;
        }
    }

    private static void appendSubline(StringBuilder builder, SublineNode subline) {

    }

    private class SublineNode {

    }
}