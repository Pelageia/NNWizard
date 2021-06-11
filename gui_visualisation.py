import tkinter as tkn
import tkinter.ttk as ttk
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle

class Visualisation(tkn.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        ## Панель вкладок
        top_part = ttk.Notebook(self)
        top_part.pack(fill='both')

        first_frame = ttk.Frame(top_part)
        top_part.add(first_frame, text='Загрузка данных', sticky='new')

        self.pwindow = tkn.PanedWindow(first_frame, orient="vertical")
        self.pwindow.pack(fill="both", expand=True)

        ## --- Загрузка
        self.load_tab = ttk.Frame(self.pwindow)
        self.pwindow.add(self.load_tab)
        self.model = None
        load_button = ttk.Button(self.load_tab, text='load model', command=self.load_model)
        load_button.grid(row=0, column=0, sticky='news')
        self.load_tab.grid_columnconfigure(0, weight=1)

    def load_model(self):
        self.model = tf.saved_model.load('model_titanic')
        history = pickle.load(open('model_titanic/history', "rb"))
        print(history["loss"])

        matplotlib.use('TkAgg')
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        fig.set_size_inches(10.5, 7.5)
        canvas = FigureCanvasTkAgg(fig, master=self.load_tab)
        plot_widget = canvas.get_tk_widget()

        plt.title("Losses train/validation")
        axs[0].plot(history["loss"], label="Train")
        axs[0].plot(history["val_loss"], label="Validation")
        plot_widget.grid(row=0, column=0, pady=5, padx=5)

        plt.title("Accuracies train/validation")
        axs[1].plot(history["accuracy"], label="Train")
        axs[1].plot(history["val_accuracy"], label="Validation")
        plot_widget.grid(row=1, column=0, pady=5, padx=5)
        plt.legend()
        plt.show()