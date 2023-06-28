package snippet_splitter_out.ds_9$gc;
public class ds_9$gc_snip_3_apply211 {
// SNIPPET_END_3
// Snippet 2
// org.unicase.dashboard.impl.NotificationOperationImpl.apply
// http://unicase.googlecode.com/svn/trunk/core/org.unicase.dashboard/src/org/unicase/dashboard/impl/Notif
// icationOperationImpl.java
// S2_1:1 method chains, good comments
/**
 * Apply the transmitted notifications (“nStore”) to the project so that
 * acknowledged notifications are deleted and other ones added.
 */
// SNIPPET_STARTS_1
public void apply211(Project project, NotificationStore nStore) {
    ProjectSpace projectSpace = project.eContainer();
    /* If project space has not been initialized, there is nothing to do. */
    if (projectSpace != null) {
        /* Get project properties and check if there are notifications. */
        PropertyManager manager = projectSpace.getPropertyManager();
        StoreProperty property = manager.getLocalProperty(NOTIFICATION_COMPOSITE);
        if (property != null) {
            Value value = property.getValue();
            /* If the project already has notifications */
            /* and if transmitted notifications are acknowledged, */
            /* then remove the transmitted notifications from the project. */
            /* Otherwise, add the transmitted notifications to the project. */
            if (value != null && value instanceof NotificationComposite) {
                NotificationComposite nComposite = value;
                if (nStore.isAcknowledged()) {
                    nComposite.getNotifications().removeAll(nStore.getNotifications());
                } else {
                    nComposite.getNotifications().addAll(nStore.getNotifications());
                }
            }
        } else {
            /* If the project did not have notifications yet */
            /* and if transmitted notifications are not acknowledged, */
            /* then add the transmitted notifications to the project */
            /* and store them in the NOTIFICATION_COMPOSITE property. */
            if (!nStore.isAcknowledged()) {
                NotificationComposite nComposite = Factory.createNotificationComposite();
                nComposite.getNotifications().addAll(nStore.getNotifications());
                manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
            }
        }
    }
}
}