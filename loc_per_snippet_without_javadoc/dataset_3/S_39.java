    private Element deliverGift(Element element) {
        Element unitElement = Message.getChildElement(element, Unit.getXMLElementTagName());

        Unit unit = (Unit) getGame().getFreeColGameObject(unitElement.getAttribute("ID"));
        unit.readFromXMLElement(unitElement);
        return unitElement;                                                                                 /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation
