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
