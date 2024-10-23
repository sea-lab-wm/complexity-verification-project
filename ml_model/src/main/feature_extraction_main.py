ROOT_PATH="/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model"

import Scalabrino_Features as scalabrino
import create_nm_data as nm
import create_itid_data as itid
import feature_merger as fm

## 1. First execute Parser.java ##
## Then execute Scalabrino_Features.java ##

## 2.run Scalabrino_Features.py to get NMI(avg) and NMI(max) features
scalabrino.main(ROOT_PATH)

# ## 3.run create_itid_data.py to get ITID(avg) and ITID(max) features
itid.main(ROOT_PATH)

## 4. run create_nm_date.py to get NM(avg) and NM(max) features
nm.main(ROOT_PATH)

## 5. feature_merger.py to merge all the features
fm.main(ROOT_PATH)