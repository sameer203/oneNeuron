from utils.model import Perceptron
from utils.all_utils import save_model, save_plot
from utils.all_utils import prepare_data
import numpy as np
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs"), level=logging.INFO, format=logging_str, filemode="a")

def main(data, eta, epochs,filename, plotfilename):

  
    df = pd.DataFrame(data)

    logging.info(f"Thi is the actual Dataframe{df}")

    X, y = prepare_data(df)

    model_OR = Perceptron(eta=eta, epochs=epochs)
    model_OR.fit(X, y)

    _ = model_OR.total_loss()

    save_model(model_OR, filename=filename)
    save_plot(df, plotfilename, model_OR)

if __name__ == "__main__": # << entry point

    OR = {
    "x1" : [0,0,1,1],
    "x2" : [0,1,0,1],
    "y" : [0,1,1,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    try:
        logging.info("\n>>>>>>> Starting training <<<<<<<")
        main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model", plotfilename="or.png")
        logging.info(">>>>>>> training Done sucessfully <<<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise(e)