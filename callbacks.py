import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback

class PredictionPlotter(Callback):
    def __init__(self, model, X_val, y_val, image_datagen, label_datagen, seed=None):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.image_datagen = image_datagen
        self.label_datagen = label_datagen
        self.seed = seed

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X_val)
        binary_predictions = (predictions > 0.25).astype(np.uint8)

        fig, axs = plt.subplots(3, 9, figsize=(15, 9))
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                if idx >= len(self.X_val):
                    continue
                axs[i, 3*j].imshow(self.X_val[idx])
                axs[i, 3*j+1].imshow(self.y_val[idx].squeeze(), cmap='gray')
                axs[i, 3*j+2].imshow(binary_predictions[idx].squeeze(), cmap='gray')
                axs[i, 3*j].axis('off')
                axs[i, 3*j+1].axis('off')
                axs[i, 3*j+2].axis('off')
        plt.tight_layout()
        plt.savefig("predictions.png")
        plt.close()
