from utils.model import Perceptron
from utils.all_utils import save_model, save_plot
from utils.all_utils import prepare_data
import numpy as np
import pandas as pd


def main(data, eta, epochs,filename, plotfilename):

  
    df = pd.DataFrame(data)

    print(df)

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

    main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model", plotfilename="or.png")