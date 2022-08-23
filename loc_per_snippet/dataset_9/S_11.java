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
