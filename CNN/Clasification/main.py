import tensorflow as tk

#classification model

def generate_model():


   model = tk.keras.Sequential([
    #first Convolutional layer
    tk.keras.layers.Conv2D(32, filter_size=3, activation="relu"),
    tk.keras.layers.MaxPool2D(pool_size=2, strides=2),

    #second
    tk.keras.layers.Conv2D(62, filter_size=3, activation="relu"),
    tk.keras.layers.MaxPool2D(pool_size=2, strides=2),

    tk.keras.layers.Flatten(),
    tk.keras.layers.Dense(1024, activation = "relu"),
    tk.keras.layers.Dense(10, activation = "softmax")
    ])
   return model
