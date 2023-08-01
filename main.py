# main.py

import cnn_model

if __name__ == "__main__":
    cnn = cnn_model.build_cnn()
    cnn.summary()
    # Here you can add code to load data, train the model, etc.
