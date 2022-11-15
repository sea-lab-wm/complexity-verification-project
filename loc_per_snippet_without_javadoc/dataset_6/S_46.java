  public static int getMinMax(Instances inst, int attrIndex, double[] minMax)
          throws Exception {
    double min = Double.NaN;
    double max = Double.NaN;
    Instance instance = null;
    int numNotMissing = 0;
    if ((minMax == null) || (minMax.length < 2)) {
      throw new Exception("Error in Program, privat method getMinMax");
    }

    Enumeration<Instance> enumInst = inst.enumerateInstances();
    if (enumInst.hasMoreElements()) {
      do {
        instance = enumInst.nextElement();
      } while (instance.isMissing(attrIndex) && (enumInst.hasMoreElements()));

      // add values if not missing
      if (!instance.isMissing(attrIndex)) {
        numNotMissing++;
        min = instance.value(attrIndex);
        max = instance.value(attrIndex);
      }
      while (enumInst.hasMoreElements()) {
        instance = enumInst.nextElement();
        if (!instance.isMissing(attrIndex)) {
          numNotMissing++;
          if (instance.value(attrIndex) < min) {
            min = (instance.value(attrIndex));
          } else {
            if (instance.value(attrIndex) > max) {
              max = (instance.value(attrIndex));
            }
          }
        }
      }
    }
    minMax[0] = min;
    minMax[1] = max;
    return numNotMissing;
  }
