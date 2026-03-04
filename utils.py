import numpy as np
import os 
import pandas as pd

def get_accuracy(y_true, y_pred):
    """
    Compute accuracy of the prediction

    :param y_true: True labels
    :param y_pred: Predicted labels
    """
    return np.mean(y_true == y_pred) * 100


def generate_submission(predictions,folder_name='submissions',model_name ="final", filename='Yte_pred.csv'):
    """
    Generate a submission file

    :param predictions: Prediction on the test set
    :param folder_name: name of the folder to put the submission in
    :param model_name: name of the model used to predict
    :param filename: name of the file containing the predictions
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file_name = model_name + filename
    file_path = os.path.join(folder_name,file_name)

    df = pd.DataFrame({'Prediction': predictions})
    df.index += 1
    df.to_csv(file_path, index_label='Id')
