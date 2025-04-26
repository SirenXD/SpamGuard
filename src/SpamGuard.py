import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from retvec.tf import RETVecTokenizer
from retvec.tf.utils import tf_cap_memory
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
    
class SpamGuard:
    '''
    SpamGuard is a class that is used to train, test, and make predictions with a model for email spam detection.
    Attributes:
        model: The model that is used to make predictions.
        CLASSES (arr): The classes that the model can predict.
        NUM_CLASSES (int): The number of classes that the model can predict.
        verbose (bool): Whether or not to print out debug information and plots to the console.
    '''

    def __init__(self, verbose=False, CLASSES=["spam", "ham"]):
        '''
        The constructor for SpamGuard class.
        '''
        self.model = None
        self.CLASSES = CLASSES
        self.NUM_CLASSES = len(self.CLASSES)
        self.verbose = verbose
        
        tf_cap_memory()

    def create_new_model(self, retvec_model_path = None):
        '''
        This function creates a new base model for the SpamGuard class that hasn't been trained.
        '''
        #Input layer
        inputs = layers.Input(shape=(1,), name='token', dtype=tf.string)
        
        #Tokenizer Layer
        x = RETVecTokenizer(model=retvec_model_path, sequence_length=2048, dropout_rate=.35, spatial_dropout_rate=.35, norm_type='batch')(inputs)

    # 2 LSTM layers with dropout and regularization
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
        x = layers.Dropout(0.4)(x)  # Dropout layer
        x = layers.Bidirectional(layers.LSTM(128, kernel_regularizer=regularizers.l2(0.001)))(x)
        x = layers.Dropout(0.4)(x)  # Dropout layer

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)  # Dropout layer

        #Output Layer
        outputs = layers.Dense(self.NUM_CLASSES, activation='sigmoid')(x)
        
        #Creates the model
        model = tf.keras.Model(inputs, outputs)
        
        #Saves it to a file
        filepath = './Models/SpamGuard_init.keras'
        model.save(filepath)

        self.model = model

        #Returns the file path so it can be loaded easily
        return filepath

    def load_model(self, model_name:str):
        '''
        This function loads a model from the file system.
        Args:
            model_name (str): The name of the model to load.
        '''
        
        self.model = tf.keras.models.load_model(f'./Models/{model_name}.keras', compile=False)

    def save_model(self, model_name:str):
        '''
        This function saves the model to the file system.
        Args:
            model_name (str): The name of the model to save.
        '''
        self.model.save(f'./Models/{model_name}.keras')

    def train(self, training_gen, validation_gen, epochs=50, export_report=False):
        '''
        This function trains the model with the given data.
        Args:
            train_data (dict): The dataset to train the model with.
            export_report (bool): Whether or not to export a PDF matplotlib graphs and the training results.
        '''

        self.model.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        lr = ReduceLROnPlateau(patience = 5, monitor = 'val_loss', factor = 0.7, verbose = 0)
        
        
        now = datetime.utcnow()
        date_str = now.strftime('%Y%m%d')
        time_str = now.strftime('%H%M%S')
        timestamp = f"{date_str}_{time_str}"
        checkpoint_filepath = f"./checkpoints/SpamGuard_{timestamp}.model.keras"
        
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_loss", mode="min", save_best_only=True) 

        history = self.model.fit(training_gen, validation_data=validation_gen, epochs=epochs, callbacks=[lr, early_stopping, model_checkpoint], workers=3)

        # Only plots the training results if verbose is set to True or export_report is set to True.
        if self.verbose or export_report:
            # Plots the training results
            fig, axs = plt.subplots(3, 2, figsize=(12, 15))

            # Plot accuracy
            axs[0, 0].plot(history.history['accuracy'])
            axs[0, 0].set_title('Accuracy')
            axs[0, 0].set_xlabel('Epoch')
            axs[0, 0].set_ylabel('Accuracy')

            # Plot validation accuracy
            axs[0, 1].plot(history.history['val_accuracy'])
            axs[0, 1].set_title('Validation Accuracy')
            axs[0, 1].set_xlabel('Epoch')
            axs[0, 1].set_ylabel('Accuracy')

            # Plot loss
            axs[1, 0].plot(history.history['loss'])
            axs[1, 0].set_title('Loss')
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].set_ylabel('Loss')

            # Plot validation loss
            axs[1, 1].plot(history.history['val_loss'])
            axs[1, 1].set_title('Validation Loss')
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('Loss')

            # Plot accuracy vs validation accuracy
            axs[2, 0].plot(history.history['accuracy'], label='Accuracy')
            axs[2, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axs[2, 0].set_title('Accuracy vs Validation Accuracy')
            axs[2, 0].set_xlabel('Epoch')
            axs[2, 0].set_ylabel('Accuracy')
            axs[2, 0].legend()

            # Plot loss vs validation loss
            axs[2, 1].plot(history.history['loss'], label='Loss')
            axs[2, 1].plot(history.history['val_loss'], label='Validation Loss')
            axs[2, 1].set_title('Loss vs Validation Loss')
            axs[2, 1].set_xlabel('Epoch')
            axs[2, 1].set_ylabel('Loss')
            axs[2, 1].legend()

            # Adjust spacing between plots
            plt.tight_layout()

            # Show the plots
            if self.verbose:
                plt.show()

            # Export the plots to a PDF, optionally
            if export_report:
                now = datetime.utcnow()
                date_str = now.strftime('%Y%m%d')
                time_str = now.strftime('%H%M%S')
                timestamp = f"{date_str}_{time_str}"
                with PdfPages(f'./reports/training_report_{timestamp}.pdf') as pdf:
                    pdf.savefig(fig)
    
    def predict(self, text:str, threshold:float=0.5):
        '''
        This function makes a prediction on whether the given text is spam or ham.
        Args:
            text (str): The text to make a prediction on.
        Returns:
            list: The prediction of the model.
        '''
        predictions = self.model(tf.constant([text], dtype=tf.string))
        out = 0
        classification = "unsure"
        probability = 0

        for i in range(self.NUM_CLASSES):
            if predictions[0][i] > threshold:
                classification = self.CLASSES[i]
                probability = round(float(predictions[0][i].numpy() * 100), 1)
                if self.verbose:
                    print(f"{classification} ({probability}%)")
                out += 1

        if not out:
            classification = "unsure"
            probability = 0

        return classification, probability