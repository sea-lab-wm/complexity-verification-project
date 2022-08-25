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
        /*HttpServletRequest h = new HttpServletRequest() {
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
        };*/

        /*logAndEmailSeriousProblemS111(new Throwable(), h);
        logAndEmailSeriousProblemS112(new Throwable(), h);
        logAndEmailSeriousProblemS113(new Throwable(), h);
        logAndEmailSeriousProblemS121(new Throwable(), h);
        logAndEmailSeriousProblemS122(new Throwable(), h);
        logAndEmailSeriousProblemS123(new Throwable(), h);*/

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

    // Snippet 2
    // org.unicase.dashboard.impl.NotificationOperationImpl.apply
    // http://unicase.googlecode.com/svn/trunk/core/org.unicase.dashboard/src/org/unicase/dashboard/impl/Notif
    // icationOperationImpl.java

    // S2_1:1 method chains, good comments
    /**
    * Apply the transmitted notifications (“nStore”) to the project so that
    * acknowledged notifications are deleted and other ones added.
    */
    //SNIPPET_STARTS
    public void apply211(Project project, NotificationStore nStore)
    {
    ProjectSpace projectSpace = project.eContainer();
    /* If project space has not been initialized, there is nothing to do. */
    if (projectSpace != null) {
    /* Get project properties and check if there are notifications. */
    PropertyManager manager = projectSpace.getPropertyManager();
    StoreProperty property =
    manager.getLocalProperty(NOTIFICATION_COMPOSITE);
    if (property != null) {
    Value value = property.getValue();
    /* If the project already has notifications */
    /* and if transmitted notifications are acknowledged, */
    /* then remove the transmitted notifications from the project. */
    /* Otherwise, add the transmitted notifications to the project. */
    if (value != null && value instanceof NotificationComposite) {
    NotificationComposite nComposite = value;
    if (nStore.isAcknowledged()) {
    nComposite.getNotifications().removeAll(
    nStore.getNotifications());
    } else {
    nComposite.getNotifications().addAll(
    nStore.getNotifications());
    }
    }
    } else {
    /* If the project did not have notifications yet */
    /* and if transmitted notifications are not acknowledged, */
    /* then add the transmitted notifications to the project */
    /* and store them in the NOTIFICATION_COMPOSITE property. */
    if (!nStore.isAcknowledged()) {
    NotificationComposite nComposite =
    Factory.createNotificationComposite();
    nComposite.getNotifications().addAll(
    nStore.getNotifications());
    manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
    }
    }
    }
    }

    // S2_1:2 method chains, bad comments
    /**
    * Apply the transmitted notifications (“nStore”) to the project so that
    * acknowledged notifications are deleted and other ones added.
    */
    //SNIPPET_STARTS
    public void apply212(Project project, NotificationStore nStore)
    {
    ProjectSpace projectSpace = project.eContainer();
    /* If projectSpace is not null */
    if (projectSpace != null) {
    /* Define local variables. */
    PropertyManager manager = projectSpace.getPropertyManager();
    StoreProperty property =
    manager.getLocalProperty(NOTIFICATION_COMPOSITE);
    if (property != null) {
    Value value = property.getValue();
    /* If value is not null and if nstore is acknowledged, */
    /* then remove nstore-notifications from notifications. */
    /* Otherwise, add nstore-notifications to notifications. */
    if (value != null && value instanceof NotificationComposite) {
    NotificationComposite nComposite = value;
    if (nStore.isAcknowledged()) {
    nComposite.getNotifications().removeAll(
    nStore.getNotifications());
    } else {
    nComposite.getNotifications().addAll(
    nStore.getNotifications());
    }
    }
    } else {
    /* If property is not null */
    /* and if nStore is not acknowledged, */
    /* then add nStore-notifications, */
    /* and set NOTIFICATION_COMPOSITE property. */
    if (!nStore.isAcknowledged()) {
    NotificationComposite nComposite =
    Factory.createNotificationComposite();
    nComposite.getNotifications().addAll(
    nStore.getNotifications());
    manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
    }
    }
    }
    }

    // S2_1:3 method chains, no comments
    /**
    * Apply the transmitted notifications (“nStore”) to the project so that
    * acknowledged notifications are deleted and other ones added.
    */
    //SNIPPET_STARTS
    public void apply213(Project project, NotificationStore nStore)
    {
    ProjectSpace projectSpace = project.eContainer();
    if (projectSpace != null) {
    PropertyManager manager = projectSpace.getPropertyManager();
    StoreProperty property =
    manager.getLocalProperty(NOTIFICATION_COMPOSITE);
    if (property != null) {
    Value value = property.getValue();
    if (value != null && value instanceof NotificationComposite) {
    NotificationComposite nComposite = value;
    if (nStore.isAcknowledged()) {
    nComposite.getNotifications().removeAll(
    nStore.getNotifications());
    } else {
    nComposite.getNotifications().addAll(
    nStore.getNotifications());
    }
    }
    } else {
    if (!nStore.isAcknowledged()) {
    NotificationComposite nComposite =
    Factory.createNotificationComposite();
    nComposite.getNotifications().addAll(
    nStore.getNotifications());
    manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
    }
    }
    }
    }

    // S2_2:1 resolved method chains, good comments
    /**
    * Apply the transmitted notifications (“nStore”) to the project so that
    * acknowledged notifications are deleted and other ones added.
    */
    //SNIPPET_STARTS
    public void apply(Project project, NotificationStore nStore)
    {
    ProjectSpace projectSpace = project.eContainer();
    /* If project space has not been initialized, there is nothing to do. */
    if (projectSpace != null) {
    /* Get project properties and check if there are notifications. */
    PropertyManager manager = projectSpace.getPropertyManager();
    StoreProperty property =
    manager.getLocalProperty(NOTIFICATION_COMPOSITE);
    NotificationList notifications;
    if (property != null) {
    Value value = property.getValue();
    /* If the project already has notifications */
    /* and if transmitted notifications are acknowledged, */
    /* then remove the transmitted notifications from the project. */
    /* Otherwise, add the transmitted notifications to the project. */
    if (value != null && value instanceof NotificationComposite) {
    NotificationComposite nComposite = value;
    notifications = nComposite.getNotifications();
    if (nStore.isAcknowledged()) {
    notifications.removeAll(nStore.getNotifications());
    } else {
    notifications.addAll(nStore.getNotifications());
    }
    }
    } else {
    /* If the project did not have notifications yet */
    /* and if transmitted notifications are not acknowledged, */
    /* then add the transmitted notifications to the project */
    /* and store them in the NOTIFICATION_COMPOSITE property. */
    if (!nStore.isAcknowledged()) {
    NotificationComposite nComposite =
    Factory.createNotificationComposite();
    notifications = nComposite.getNotifications();
    notifications.addAll(nStore.getNotifications());
    manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
    }
    }
    }
    }

    // S2_2:2 resolved method chains, bad comments
    /**
    * Apply the transmitted notifications (“nStore”) to the project so that
    * acknowledged notifications are deleted and other ones added.
    */
    //SNIPPET_STARTS
    public void apply222(Project project, NotificationStore nStore)
    {
    ProjectSpace projectSpace = project.eContainer();
    /* If projectSpace is not null */
    if (projectSpace != null) {
    /* Define local variables. */
    PropertyManager manager = projectSpace.getPropertyManager();
    StoreProperty property =
    manager.getLocalProperty(NOTIFICATION_COMPOSITE);
    NotificationList notifications;
    if (property != null) {
    Value value = property.getValue();
    /* If value is not null and if nstore is acknowledged, */
    /* then remove nstore-notifications from notifications. */
    /* Otherwise, add nstore-notifications to notifications. */
    if (value != null && value instanceof NotificationComposite) {
    NotificationComposite nComposite = value;
    notifications = nComposite.getNotifications();
    if (nStore.isAcknowledged()) {
    notifications.removeAll(nStore.getNotifications());
    } else {
    notifications.addAll(nStore.getNotifications());
    }
    }
    } else {
    /* If property is not null */
    /* and if nStore is not acknowledged, */
    /* then add nStore-notifications, */
    /* and set NOTIFICATION_COMPOSITE property. */
    if (!nStore.isAcknowledged()) {
    NotificationComposite nComposite =
    Factory.createNotificationComposite();
    notifications = nComposite.getNotifications();
    notifications.addAll(nStore.getNotifications());
    manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
    }
    }
    }
    }

    // S2_2:3 resolved method chains, no comments
    /**
    * Apply the transmitted notifications (“nStore”) to the project so that
    * acknowledged notifications are deleted and other ones added.
    */
    //SNIPPET_STARTS
    public void apply223(Project project, NotificationStore nStore)
    {
    ProjectSpace projectSpace = project.eContainer();
    if (projectSpace != null) {
    PropertyManager manager = projectSpace.getPropertyManager();
    StoreProperty property =
    manager.getLocalProperty(NOTIFICATION_COMPOSITE);
    NotificationList notifications;
    if (property != null) {
    Value value = property.getValue();
    if (value != null && value instanceof NotificationComposite) {
    NotificationComposite nComposite = value;
    notifications = nComposite.getNotifications();
    if (nStore.isAcknowledged()) {
    notifications.removeAll(nStore.getNotifications());
    } else {
    notifications.addAll(nStore.getNotifications());
    }
    }
    } else {
    if (!nStore.isAcknowledged()) {
    NotificationComposite nComposite =
    Factory.createNotificationComposite();
    notifications = nComposite.getNotifications();
    notifications.addAll(nStore.getNotifications());
    manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
    }
    }
    }
    }

    // Snippet 3
    // org.unicase.docExport.docWriter.ITextWriter.writeUTable
    // http://unicase.googlecode.com/svn/trunk/core/org.unicase.docExport/src/org/unicase/docExport/docWriter/
    // ITextWriter.java

    // S3_1:1 (Revision 1) method chains, good comments
    /**
    * The images of a unified table are copied to a simplified “yes/no” table.
    */
    //SNIPPET_STARTS
    protected Table copyUTable311(UTable uTable)
    {
    /* Create a new table with the same size and color as uTable. */
    Table table = new Table(uTable.getColumnsCount());
    table.setBorderColor(new Color (
    uTable.getBoxModel().getBorderColor().getRed(),
    uTable.getBoxModel().getBorderColor().getGreen(),
    uTable.getBoxModel().getBorderColor().getBlue()));
    /* Go through all entries of uTable and copy image information. */
    Cell cell = null; // had to be added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    UParagraph paragraph = uCell.getContent();
    /* Check if there is image information to copy. */
    if (paragraph.getChildren().size() > 0 &&
    paragraph.getChildren().get(0) instanceof UImage) {
    UImage image = paragraph.getChildren().get(0);
    /* Copy the last segment of the image. False and true */
    /* are transformed to “no” and “yes”, respectively. */
    if (image.getPath().lastSegment().startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (image.getPath().lastSegment().startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(image.getPath().lastSegment());
    }
    }
    /* Copy background color of uCell. */
    if (uCell.getBoxModel().getBackgroundColor() != null) {
    cell.setBackgroundColor(uCell.getBoxModel().getBackgroundColor());
    }
    }
    table.addCell(cell);
    }
    return table;
    }

    // S3_1:2 (Revision 1) method chains, bad comments
    /**
    * The images of a unified table are copied to a simplified “yes/no” table.
    */
    //SNIPPET_STARTS
    protected Table copyUTable312(UTable uTable)
    {
    /* Define local variables. */
    Table table = new Table(uTable.getColumnsCount());
    table.setBorderColor(new Color (
    uTable.getBoxModel().getBorderColor().getRed(),
    uTable.getBoxModel().getBorderColor().getGreen(),
    uTable.getBoxModel().getBorderColor().getBlue()));
    Cell cell = null; // Had to be added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    /* Define local variable. */
    UParagraph paragraph = uCell.getContent();
    /* If children size larger than 0 and first children is an image. */
    if (paragraph.getChildren().size() > 0 &&
    paragraph.getChildren().get(0) instanceof UImage) {
    UImage image = paragraph.getChildren().get(0);
    /* If segment starts with false, then create a cell with “no”. */
    /* Otherwise, if segment starts with true, then a create cell */
    /* with “yes”. Otherwise create cell with segment. */
    if (image.getPath().lastSegment().startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (image.getPath().lastSegment().startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(image.getPath().lastSegment());
    }
    }
    /* Copy background color. */
    if (uCell.getBoxModel().getBackgroundColor() != null) {
    cell.setBackgroundColor(uCell.getBoxModel().getBackgroundColor());
    }
    }
    table.addCell(cell);
    }
    return table;
    }

    // S3_1:3 (Revision 1) method chains, no comments
    /**
    * The images of a unified table are copied to a simplified “yes/no” table.
    */
    //SNIPPET_STARTS
    protected Table copyUTable313(UTable uTable)
    {
    Table table = new Table(uTable.getColumnsCount());
    table.setBorderColor(new Color (
    uTable.getBoxModel().getBorderColor().getRed(),
    uTable.getBoxModel().getBorderColor().getGreen(),
    uTable.getBoxModel().getBorderColor().getBlue()));
    Cell cell = null; // Had to be added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    UParagraph paragraph = uCell.getContent();
    if (paragraph.getChildren().size() > 0 &&
    paragraph.getChildren().get(0) instanceof UImage) {
    UImage image = paragraph.getChildren().get(0);
    if (image.getPath().lastSegment().startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (image.getPath().lastSegment().startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(image.getPath().lastSegment());
    }
    }
    if (uCell.getBoxModel().getBackgroundColor() != null) {
    cell.setBackgroundColor(uCell.getBoxModel().getBackgroundColor());
    }
    }
    table.addCell(cell);
    }
    return table;
    }

    // S3_2:1 (Revision 1) Fully resolved method chains, good comments
    /**
    * The images of a unified table are copied to a simplified “yes/no” table.
    */
    //SNIPPET_STARTS
    protected Table writeUTable321(UTable uTable)
    {
    /* Create a new table with the same size and color as uTable. */
    Table table = new Table(uTable.getColumnsCount());
    BoxModelOption option = uTable.getBoxModel();
    UColor color = option.getBorderColor();
    table.setBorderColor(new Color
    (color.getRed(),color.getGreen(),color.getBlue()));
    /* Go through all entries of uTable and copy image information. */
    Cell cell = null; // Added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    UParagraph paragraph = uCell.getContent();
    UChildren children = paragraph.getChildren();
    /* Check if there is image information to copy. */
    if (children.size() > 0 && children.get(0) instanceof UImage) {
    UImage image = children.get(0);
    Path path = image.getPath();
    USegment segment = path.lastSegment();
    /* Copy the last segment of the image. False and true */
    /* are transformed to “no” and “yes”, respectively. */
    if (segment.startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (segment.startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(segment);
    }
    }
    /* Copy background color of uCell. */
    option = uCell.getBoxModel();
    color = option.getBackgroundColor();
    if (color != null) {
    cell.setBackgroundColor(color);
    }
    }
    table.addCell(cell);
    }
    return table;
    }

    // S3_2:2 (Revision 1) Fully resolved method chains, bad comments
    /**
    * The images of a unified table are copied to a simplified “yes/no” table.
    */
    //SNIPPET_STARTS
    protected Table writeUTable(UTable uTable)
    {
    /* Define local variables. */
    Table table = new Table(uTable.getColumnsCount());
    BoxModelOption option = uTable.getBoxModel();
    UColor color = option.getBorderColor();
    table.setBorderColor(new Color
    (color.getRed(),color.getGreen(),color.getBlue()));
    Cell cell = null; // Added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    /* Define local variable. */
    UParagraph paragraph = uCell.getContent();
    UChildren children = paragraph.getChildren();
    /* If children size larger than 0 and first children is an image. */
    if (children.size() > 0 && children.get(0) instanceof UImage) {
    UImage image = children.get(0);
    Path path = image.getPath();
    USegment segment = path.lastSegment();
    /* If segment starts with false, then create a cell with “no”. */
    /* Otherwise, if segment starts with true, then a create cell */
    /* with “yes”. Otherwise create cell with segment. */
    if (segment.startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (segment.startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(segment);
    }
    }
    /* Copy background color. */
    option = uCell.getBoxModel();
    color = option.getBackgroundColor();
    if (color != null) {
    cell.setBackgroundColor(color);
    }
    }
    table.addCell(cell);
    }
    return table;
    }

    // S3_2:3 (Revision 1) Fully resolved method chains, no comments
    /**
    * The images of a unified table are copied to a simplified “yes/no” table.
    */
    //SNIPPET_STARTS
    protected Table writeUTable323(UTable uTable)
    {
    Table table = new Table(uTable.getColumnsCount());
    BoxModelOption option = uTable.getBoxModel();
    UColor color = option.getBorderColor();
    table.setBorderColor(new Color
    (color.getRed(),color.getGreen(),color.getBlue()));
    Cell cell = null; // Added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    UParagraph paragraph = uCell.getContent();
    UChildren children = paragraph.getChildren();
    if (children.size() > 0 && children.get(0) instanceof UImage) {
    UImage image = children.get(0);
    Path path = image.getPath();
    USegment segment = path.lastSegment();
    if (segment.startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (segment.startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(segment);
    }
    }
    option = uCell.getBoxModel();
    color = option.getBackgroundColor();
    if (color != null) {
    cell.setBackgroundColor(color);
    }
    }
    table.addCell(cell);
    }
    return table;
    }

    // Snippet 4
    // raptor.chess.pgn.PgnUtils.getMove
    // http://code.google.com/p/raptor‐chess‐interface/source/browse/tags/v.92/src/raptor/chess/pgn/PgnUtils.jav
    // a

    // S4_1:1 method chains, good comments
    /**
    * Generates a string (“builder”) for a given chess move in PGN (Portable
    * Game Notation). This includes the move number and all NAG annotations.
    */
    //SNIPPET_STARTS
    public static boolean getMove411(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    /* Add move number and move details. */
    builder.append(moveNumber).append(move.isWhitesMove() ? ". " : "... ").
    append(move.toString());
    /* Add all available analysis data (sublines). */
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    /* Add all text comments of the move. */
    for (Comment comment : move.getComments()) {
    builder.append(" {").append(comment.getText()).append("}");
    }
    /* Add all Numeric Annotations Glyphs (NAGs) of the move. */
    for (Nag nag : move.getNags()) {
    builder.append(" ").append(nag.getNagString());
    }
    /* Add time taken for the move. */
    builder.append(" {").append(move.getTimeTakenForMove().getText()).append("}");
    return result;
    }

    // S4_1:2 method chains, bad comments
    /**
    * Generates a string (“builder”) for a given chess move in PGN (Portable
    * Game Notation). This includes the move number and all NAG annotations.
    */
    //SNIPPET_STARTS
    public static boolean getMove412(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    /* Add move data. */
    builder.append(moveNumber).append(move.isWhitesMove() ? ". " : "... ").
    append(move.toString());
    /* Add sublines. */
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    /* Add comments. */
    for (Comment comment : move.getComments()) {
    builder.append(" {").append(comment.getText()).append("}");
    }
    /* Add NAGs. */
    for (Nag nag : move.getNags()) {
    builder.append(" ").append(nag.getNagString());
    }
    /* Add time. */
    builder.append(" {").append(move.getTimeTakenForMove().getText()).append("}");
    return result;
    }

    // S4_1:3 method chains, no comments
    /**
    * Generates a string (“builder”) for a given chess move in PGN (Portable
    * Game Notation). This includes the move number and all NAG annotations.
    */
    //SNIPPET_STARTS
    public static boolean getMove413(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    builder.append(moveNumber).append(move.isWhitesMove() ? ". " : "... ").
    append(move.toString());
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    for (Comment comment : move.getComments()) {
    builder.append(" {").append(comment.getText()).append("}");
    }
    for (Nag nag : move.getNags()) {
    builder.append(" ").append(nag.getNagString());
    }
    builder.append(" {").append(move.getTimeTakenForMove().getText()).append("}");
    return result;
    }

    // S4_2:1 resolved method chains, good comments
    /**
    * Generates a string (“builder”) for a given chess move in PGN (Portable
    * Game Notation). This includes the move number and all NAG annotations.
    */
    //SNIPPET_STARTS
    public static boolean getMove421(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    /* Add move number and move details. */
    builder.append(moveNumber);
    builder.append(move.isWhitesMove() ? ". " : "... ");
    builder.append(move.toString());
    /* Add all available analysis data (sublines). */
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    /* Add all text comments of the move. */
    for (Comment comment : move.getComments()) {
    builder.append(" {");
    builder.append(comment.getText());
    builder.append("}");
    }
    /* Add all Numeric Annotations Glyphs (NAGs) of the move. */
    for (Nag nag : move.getNags()) {
    builder.append(" ");
    builder.append(nag.getNagString());
    }
    /* Add time taken for the move. */
    builder.append(" {");
    TimeTakenForMove timeTaken = move.getTimeTakenForMove();
    builder.append(timeTaken.getText());
    builder.append("}");
    return result;
    }

    // S4_2:2 resolved method chains, bad comments
    /**
    * Generates a string (“builder”) for a given chess move in PGN (Portable
    * Game Notation). This includes the move number and all NAG annotations.
    */
    //SNIPPET_STARTS
    public static boolean getMove422(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    /* Add move data. */
    builder.append(moveNumber);
    builder.append(move.isWhitesMove() ? ". " : "... ");
    builder.append(move.toString());
    /* Add sublines. */
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    /* Add comments. */
    for (Comment comment : move.getComments()) {
    builder.append(" {");
    builder.append(comment.getText());
    builder.append("}");
    }
    /* Add NAGs. */
    for (Nag nag : move.getNags()) {
    builder.append(" ");
    builder.append(nag.getNagString());
    }
    /* Add time. */
    builder.append(" {");
    TimeTakenForMove timeTaken = move.getTimeTakenForMove();
    builder.append(timeTaken.getText());
    builder.append("}");
    return result;
    }

    // S4_2:3 resolved method chains, no comments
    /**
    * Generates a string (“builder”) for a given chess move in PGN (Portable
    * Game Notation). This includes the move number and all NAG annotations.
    */
    //SNIPPET_STARTS
    public static boolean getMove(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    builder.append(moveNumber);
    builder.append(move.isWhitesMove() ? ". " : "... ");
    builder.append(move.toString());
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    for (Comment comment : move.getComments()) {
    builder.append(" {");
    builder.append(comment.getText());
    builder.append("}");
    }
    for (Nag nag : move.getNags()) {
    builder.append(" ");
    builder.append(nag.getNagString());
    }
    builder.append(" {");
    TimeTakenForMove timeTaken = move.getTimeTakenForMove();
    builder.append(timeTaken.getText());
    builder.append("}");
    return result;
    }

    // Snippet 5
    // org.eclipse.jface.dialogs.Dialog.shortenText
    // http://www.docjar.com/html/api/org/eclipse/jface/dialogs/Dialog.java.html

    // S5_1:1 method chains, good comments
    /**
    * Shortens the given text textValue so that its width in pixels does
    * not exceed the width of the given control. To do that, shortenText
    * overrides as many as necessary characters in the center of the original
    * text with an ellipsis character (constant ELLIPSIS = "...").
    */
    //SNIPPET_STARTS
    public static String shortenText511(String textValue, Control control)
    {
    GraphicsContext gc = new GraphicsContext(control);
    int maxExtent = gc.textExtent(textValue).x;
    int maxWidth = control.getBounds().width - 5;
    /* Set start and end points for the center of the text. */
    int length = textValue.length();
    int start = length/2;
    int end = length/2 + 1;
    /* Take away more and more characters in the center of textValue */
    /* and replace them by a single ellipsis character until it is */
    /* short enough. */
    while (start >= 0 && end < length) {
    String s1 = textValue.substring(0, start);
    String s2 = textValue.substring(end, length);
    String s = s1 + ELLIPSIS + s2;
    int l = gc.textExtent(s).x;
    /* When the text fits, we stop and return the shortened string. */
    if (l < maxWidth) {
    gc.dispose();
    return s;
    }
    start--;
    end++;
    }
    return null; // Had to be added to allow compilation
    }

    // S5_1:2 method chains, bad comments
    /**
    * Shortens the given text textValue so that its width in pixels does
    * not exceed the width of the given control. To do that, shortenText
    * overrides as many as necessary characters in the center of the original
    * text with an ellipsis character (constant ELLIPSIS = "...").
    */
    //SNIPPET_STARTS
    public static String shortenText512(String textValue, Control control)
    {
    /* Define local variables. */
    GraphicsContext gc = new GraphicsContext(control);
    int maxExtent = gc.textExtent(textValue).x;
    int maxWidth = control.getBounds().width - 5;
    /* Define further local variables. */
    int length = textValue.length();
    int start = length/2;
    int end = length/2 + 1;
    /* While the condition holds true, */
    /* override characters between s1 and s2 with ELLIPSIS. */
    while (start >= 0 && end < length) {
    String s1 = textValue.substring(0, start);
    String s2 = textValue.substring(end, length);
    String s = s1 + ELLIPSIS + s2;
    int l = gc.textExtent(s).x;
    /* When l is smaller than maxWidth, return s. */
    if (l < maxWidth) {
    gc.dispose();
    return s;
    }
    start--;
    end++;
    }
    return null; // Had to be added to allow compilation
    }

// S5_1:3 method chains, no comments
    /**
    * Shortens the given text textValue so that its width in pixels does
    * not exceed the width of the given control. To do that, shortenText
    * overrides as many as necessary characters in the center of the original
    * text with an ellipsis character (constant ELLIPSIS = "...").
    */
    //SNIPPET_STARTS
    public static String shortenText513(String textValue, Control control)
    {
    GraphicsContext gc = new GraphicsContext(control);
    int maxExtent = gc.textExtent(textValue).x;
    int maxWidth = control.getBounds().width - 5;
    int length = textValue.length();
    int start = length/2;
    int end = length/2 + 1;
    while (start >= 0 && end < length) {
    String s1 = textValue.substring(0, start);
    String s2 = textValue.substring(end, length);
    String s = s1 + ELLIPSIS + s2;
    int l = gc.textExtent(s).x;
    if (l < maxWidth) {
    gc.dispose();
    return s;
    }
    start--;
    end++;
    }
    return null; // Had to be added to allow compilation
    }

    // S5_2:1 resolved method chains, good comments
    /**
    * Shortens the given text textValue so that its width in pixels does
    * not exceed the width of the given control. To do that, shortenText
    * overrides as many as necessary characters in the center of the original
    * text with an ellipsis character (constant ELLIPSIS = "...").
    */
    //SNIPPET_STARTS
    public static String shortenText521(String textValue, Control control)
    {
    GraphicsContext gc = new GraphicsContext(control);
    Extent extent = gc.textExtent(textValue);
    int maxExtent = extent.x;
    Bounds bounds = control.getBounds();
    int maxWidth = bounds.width - 5;
    /* Set start and end points for the center of the text. */
    int length = textValue.length();
    int start = length/2;
    int end = length/2 + 1;
    /* Take away more and more characters in the center of textValue */
    /* and replace them by a single ellipsis character until it is */
    /* short enough. */
    while (start >= 0 && end < length) {
    String s1 = textValue.substring(0, start);
    String s2 = textValue.substring(end, length);
    String s = s1 + ELLIPSIS + s2;
    extent = gc.textExtent(s);
    int l = extent.x;
    /* When the text fits, we stop and return the shortened string. */
    if (l < maxWidth) {
    gc.dispose();
    return s;
    }
    start--;
    end++;
    }
    return null; // Had to be added to allow compilation
    }

    // S5_2:2 resolved method chains, bad comments
    /**
    * Shortens the given text textValue so that its width in pixels does
    * not exceed the width of the given control. To do that, shortenText
    * overrides as many as necessary characters in the center of the original
    * text with an ellipsis character (constant ELLIPSIS = "...").
    */
    //SNIPPET_STARTS
    public static String shortenText522(String textValue, Control control)
    {
    /* Define local variables. */
    GraphicsContext gc = new GraphicsContext(control);
    Extent extent = gc.textExtent(textValue);
    int maxExtent = extent.x;
    Bounds bounds = control.getBounds();
    int maxWidth = bounds.width - 5;
    /* Define further local variables. */
    int length = textValue.length();
    int start = length/2;
    int end = length/2 + 1;
    /* While the condition holds true, */
    /* override characters between s1 and s2 with ELLIPSIS. */
    while (start >= 0 && end < length) {
    String s1 = textValue.substring(0, start);
    String s2 = textValue.substring(end, length);
    String s = s1 + ELLIPSIS + s2;
    extent = gc.textExtent(s);
    int l = extent.x;
    /* When l is smaller than maxWidth, return s. */
    if (l < maxWidth) {
    gc.dispose();
    return s;
    }
    start--;
    end++;
    }
    return null; // Had to be added to allow compilation
    }

    // S5_2:3 resolved method chains, no comments
    /**
    * Shortens the given text textValue so that its width in pixels does
    * not exceed the width of the given control. To do that, shortenText
    * overrides as many as necessary characters in the center of the original
    * text with an ellipsis character (constant ELLIPSIS = "...").
    */
    //SNIPPET_STARTS
    public static String shortenText523(String textValue, Control control)
    {
    GraphicsContext gc = new GraphicsContext(control);
    Extent extent = gc.textExtent(textValue);
    int maxExtent = extent.x;
    Bounds bounds = control.getBounds();
    int maxWidth = bounds.width - 5;
    int length = textValue.length();
    int start = length/2;
    int end = length/2 + 1;
    while (start >= 0 && end < length) {
    String s1 = textValue.substring(0, start);
    String s2 = textValue.substring(end, length);
    String s = s1 + ELLIPSIS + s2;
    extent = gc.textExtent(s);
    int l = extent.x;
    if (l < maxWidth) {
    gc.dispose();
    return s;
    }
    start--;
    end++;
    }
    return null; // Had to be added to allow compilation
    }
    //SNIPPETS_END

    private static class Move {
        public int getFullMoveCount() {
            return 0;
        }

        public boolean isWhitesMove() {
            return false;
        }

        public SublineNode[] getSublines() {
            return new SublineNode[0];
        }


        public Comment[] getComments() {
            return new Comment[0];
        }


        public Nag[] getNags() {
            return new Nag[0];
        }

        public TimeTakenForMove getTimeTakenForMove() {
            return null;
        }
    }

    private static class TimeTakenForMove{
        public char[] getText() {
            return new char[0];
        }
    }

    private static class Control {
        public Bounds getBounds() {
            return null;
        }
    }

    private static class GraphicsContext {
        public GraphicsContext(Control control) {
        }

        public Extent textExtent(String textValue) {
            return null;
        }

        public void dispose() {

        }
    }

    private static class Extent {
        public int x;
    }

    private static class Bounds {
        public int width;
    }

    private class Nag {
        public char[] getNagString() {
            return new char[0];
        }
    }

    private class Comment {
        public char[] getText() {
            return new char[0];
        }
    }

    private class SublineNode {

    }

    private class Project {
        public ProjectSpace eContainer() {
            return null;
        }
    }

    private class ProjectSpace {
        public PropertyManager getPropertyManager() {
            return null;
        }
    }

    private class NotificationStore {
        public boolean isAcknowledged() {
            return false;
        }

        public Collection getNotifications() {
            return null;
        }
    }

    private class PropertyManager {
        public StoreProperty getLocalProperty(String notificationComposite) {
            return null;
        }

        public void setLocalProperty(String notificationComposite, NotificationComposite nComposite) {

        }
    }

    private class StoreProperty {
        public Value getValue() {
            return null;
        }
    }

    private class Value extends NotificationComposite{
    }

    private class NotificationComposite {
        public <E> NotificationList getNotifications() {
            return null;
        }
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

    private class UTable {
        public Object getColumnsCount() {
            return null;
        }

        public BoxModelOption getBoxModel() {
            return null;
        }

        public UTableCell[] getEntries() {
            return new UTableCell[0];
        }
    }

    private class Table {
        public Table(Object columnsCount) {

        }

        public void setBorderColor(Color color) {

        }

        public void addCell(Cell cell) {

        }
    }

    private class BoxModelOption {
        public UColor getBorderColor() {
            return null;
        }

        public UColor getBackgroundColor() {
            return null;
        }
    }

    private class Cell {
        public Cell(USegment no) {

        }

        public Cell(String no) {

        }

        public void setBackgroundColor(UColor backgroundColor) {

        }
    }

    private class UTableCell {
        public UParagraph getContent() {
            return null;
        }

        public BoxModelOption getBoxModel() {
            return null;
        }
    }

    private class UParagraph {
        public UChildren getChildren() {
            return null;
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
}