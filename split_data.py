#  ===========================================================================
#  @file:   split_data.py
#  @brief:  Separate data into three sets (Train, validation and train)
#  @author: Johan Mendez
#  @date:   19/05/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================
from sklearn.model_selection import train_test_split


def split_stratified(X, y, frac_train=0.6, frac_test=0.25, 
                     frac_val=0.15, random_state=11):

  '''
  
  '''

  if frac_train + frac_test + frac_val != 1.0:
    raise ValueError("Las fracciones %f, %f, %f no suman 1.0")

  # Dividir datos originales en variables temporales y datos de entrenamiento
  X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=(1.0-frac_train),
                                                    random_state=random_state
                                                    )

  # Dividir datos temporal en validación y test
  rel_test = frac_test/(frac_val + frac_test)
  X_val , X_test, y_val, y_test = train_test_split(X_temp, 
                                                  y_temp,
                                                  stratify=y_temp,
                                                  test_size=rel_test,
                                                  random_state=random_state)

  assert len(X) == len(X_train) + len(X_test) + len(X_val)

  return X_train, X_test, X_val, y_train, y_test, y_val

