from oneNeuron.perceptron import Perceptron
from utilities.all_utils import save_model, save_plot, prepare_data
#from utils.model import Perceptron
#from utils.all_utils import save_model, save_plot
#from utils.all_utils import prepare_data
import numpy as np
import pandas as pd
import os
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs_and"), level=logging.INFO, format=logging_str)


def main(data, eta, epochs,filename, plotfilename):

  
    df = pd.DataFrame(data)

    print(df)

    X, y = prepare_data(df)

    model_AND = Perceptron(eta=eta, epochs=epochs)
    model_AND.fit(X, y)

    _ = model_AND.total_loss()

    save_model(model_AND, filename=filename)
    save_plot(df, plotfilename, model_AND)

if __name__ == "__main__": # << entry point

    AND = {
    "x1" : [0,0,1,1],
    "x2" : [0,1,0,1],
    "y" : [0,0,0,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    main(data=AND, eta=ETA, epochs=EPOCHS, filename="and_new.model", plotfilename="and_new.png")