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
