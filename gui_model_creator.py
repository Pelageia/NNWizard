import tkinter.ttk as ttk
import tkinter as tkn
from pandastable import Table, TableModel
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
# import keras as k
from tkinter import Canvas
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModelCreator(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        ## Панель вкладок
        top_part = ttk.Notebook(self)
        top_part.pack(fill='both')

        first_frame = ttk.Frame(top_part)
        top_part.add(first_frame, text='Разбиение данных', sticky='news')

        self.pwindow = tkn.PanedWindow(first_frame, orient="vertical")
        self.pwindow.pack(fill="both", expand=True)

        # # --- Загрузка
        load_tab = ttk.Frame(self.pwindow)
        self.pwindow.add(load_tab)

        load_button = ttk.Button(load_tab, text='load pickles', command=self.load_pickle)
        load_button.grid(row=0, column=0, sticky='news')
        load_tab.grid_columnconfigure(0, weight=1)
        #
        Label = ttk.Label(load_tab, text="Choose column for filtration")
        Label.grid(row=0, column=1, sticky='new')
        load_tab.grid_columnconfigure(1, weight=1)

        self.procent_entry = ttk.Entry(load_tab)
        self.procent_entry.grid(row=0, column=2, sticky='news')
        load_tab.grid_columnconfigure(2, weight=1)

        load_button = ttk.Button(load_tab, text='Разделение', command=self.count_train)
        load_button.grid(row=0, column=3, sticky='news')
        load_tab.grid_columnconfigure(3, weight=1)
        #
        second_frame = ttk.Frame(self.pwindow)
        self.pwindow.add(second_frame)

        self.stk_count = 0
        self.stk_count_test = 0
        df = None
        self.current_table = Table(second_frame, dataframe=
        pd.DataFrame([['Training', 100, self.stk_count], ['Test', 0, self.stk_count_test]],
                     columns=['Dataset', 'Size %', 'Size str']),
                                   showtoolbar=0, showstatusbar=1)
        self.current_table.enable_menus = False
        self.current_table.grid(row=0, column=0, sticky='news')
        #

        self.current_table.show()
        self.current_table.redraw()
        #
        clean_tab = ttk.Frame(top_part)
        top_part.add(clean_tab, text='Создание модели', sticky='news')

        self.pwindow_separating = tkn.PanedWindow(clean_tab, orient="vertical")
        self.pwindow_separating.pack(fill="both", expand=True)

        frame = ttk.LabelFrame(self.pwindow_separating, text='Network name')
        self.pwindow_separating.add(frame)

        self.network_name_entry = ttk.Entry(frame)
        self.network_name_entry.grid(row=0, column=0, pady=5, padx=5, sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)


        frame = ttk.LabelFrame(self.pwindow_separating, text='Network Type')
        self.pwindow_separating.add(frame)

        network_types = ["Sequential"]
        self.Combobox_nt = ttk.Combobox(frame, values=network_types, width=40, validate='key')
        self.Combobox_nt.current(0)
        self.Combobox_nt.grid(row=0, column=0, pady=5, padx=5, sticky='new')
        clean_tab.grid_columnconfigure(0, weight=1)
        clean_tab.grid_rowconfigure(0, weight=1)

        frame = ttk.LabelFrame(self.pwindow_separating, text='Learning Algorithm:')
        self.pwindow_separating.add(frame)

        learning_algs = ["sgd", "adam"]
        self.Combobox_la = ttk.Combobox(frame, values=learning_algs, width=40, validate='key')
        self.Combobox_la.grid(row=0, column=0, pady=5, padx=5, sticky='new')
        self.Combobox_la.current(0)
        frame = ttk.LabelFrame(self.pwindow_separating, text='Loss Function')
        self.pwindow_separating.add(frame)

        #
        loss_funcs = ["mse", "sparse_categorical_crossentropy"]

        self.Combobox_lf = ttk.Combobox(frame, values=loss_funcs, width=40, validate='key')
        self.Combobox_lf.grid(row=0, column=0, pady=5, padx=5, sticky='new')
        self.Combobox_lf.current(0)
        clean_tab.grid_columnconfigure(0, weight=1)

        frame = ttk.LabelFrame(self.pwindow_separating, text='Metrics Function')
        self.pwindow_separating.add(frame)
        #
        metrics_funcs = ["accuracy"]
        self.Combobox_mf = ttk.Combobox(frame, values=metrics_funcs, width=40, validate='key')
        self.Combobox_mf.grid(row=0, column=0, pady=5, padx=5, sticky='new')
        self.Combobox_mf.current(0)
        clean_tab.grid_columnconfigure(0, weight=1)

        frame = ttk.LabelFrame(self.pwindow_separating, text='Number of layers')
        self.pwindow_separating.add(frame)


        self.number_layers_entry = ttk.Entry(frame)
        self.number_layers_entry.insert(0, "2")
        self.number_layers_entry.grid(row=0, column=0, pady=5, padx=5, sticky='news')

        clean_tab.grid_columnconfigure(0, weight=1)

        model_create_layers_button = ttk.Button(frame, text='Create layers', command=self.create_layers)
        model_create_layers_button.grid(row=0, column=1, pady=5, padx=5, sticky='news')
        clean_tab.grid_columnconfigure(1, weight=1)

        frame = ttk.LabelFrame(self.pwindow_separating, text='Properties for Layer')
        self.pwindow_separating.add(frame)
        #
        self.layers = []
        self.layers_count = []
        mas = ["0", "1"]  # дефолт 0 - вход, 1 - выход
        self.Combobox_pfl = ttk.Combobox(frame, width=40, values=mas, validate='key')
        self.Combobox_pfl.current(0)
        self.Combobox_pfl.grid(row=0, column=0, pady=5, padx=5, sticky='new')

        frame_pf = ttk.Frame(frame)
        frame_pf.grid(row=1, column=0, pady=5, padx=5, sticky='new')

        Label = ttk.Label(frame_pf, text="Number of neurons:")
        Label.grid(row=0, column=0, sticky='new')
        clean_tab.grid_columnconfigure(0, weight=1)

        self.number_neurons_entry = ttk.Entry(frame_pf)
        self.number_neurons_entry.insert(0, "3")
        self.number_neurons_entry.grid(row=0, column=1, pady=5, padx=5,  sticky='news')
        clean_tab.grid_columnconfigure(1, weight=1)

        frame_pf = ttk.Frame(frame)
        frame_pf.grid(row=2, column=0, pady=5, padx=5, sticky='new')

        Label = ttk.Label(frame_pf, text="Transfer function:")
        Label.grid(row=0, column=0, sticky='new')
        clean_tab.grid_columnconfigure(0, weight=1)
        #
        transfer_funcs = ["relu", "sigmoid", "softmax"]
        self.Combobox_transf = ttk.Combobox(frame_pf, values=transfer_funcs, width=40, validate='key')
        self.Combobox_transf.current(0)
        self.Combobox_transf.grid(row=0, column=1, pady=5, padx=5, sticky='new')

        frame_pf = ttk.Frame(frame)
        frame_pf.grid(row=3, column=0, pady=5, padx=5, sticky='new')
        change_layer_button = ttk.Button(frame_pf, text='Change Layer', command=self.change_layer)
        change_layer_button.grid(row=0, column=0, sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)

        frame = ttk.LabelFrame(self.pwindow_separating, text='')
        self.pwindow_separating.add(frame)

        model_create_button = ttk.Button(frame, text='Create model', command=self.create_model)
        model_create_button.grid(row=0, column=0, sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)


        third_frame = ttk.Frame(top_part)
        top_part.add(third_frame, text='Обучение модели', sticky='news')

        self.pwindow = tkn.PanedWindow(third_frame, orient="vertical")
        self.pwindow.pack(fill="both", expand=True)

        load_tab = ttk.LabelFrame(self.pwindow, text='Learning parametrs')
        self.pwindow.add(load_tab)

        Label = ttk.Label(load_tab, text="Epochs:")
        Label.grid(row=0, column=0,  pady=5, padx=5, sticky='new')
        load_tab.grid_columnconfigure(0, weight=1)

        self.number_epochs_entry = ttk.Entry(load_tab)
        self.number_epochs_entry.insert(0, "100")
        self.number_epochs_entry.grid(row=0, column=1, pady=5, padx=5, sticky='news')
        load_tab.grid_columnconfigure(1, weight=1)

        Label = ttk.Label(load_tab, text="Validation_split:")
        Label.grid(row=1, column=0, pady=5, padx=5, sticky='new')
        load_tab.grid_columnconfigure(0, weight=1)

        self.validation_split_entry = ttk.Entry(load_tab)
        self.validation_split_entry.insert(0, "0.2")
        self.validation_split_entry.grid(row=1, column=1, pady=5, padx=5, sticky='news')
        load_tab.grid_columnconfigure(1, weight=1)

        learn_button = ttk.Button(load_tab, text='Learn model', command=self.learn_model)
        learn_button.grid(row=2, column=0, pady=5, padx=5, sticky='news')
        load_tab.grid_columnconfigure(2, weight=1)

        self.my_model = None
        self.final_modal = None
        save_model_button = ttk.Button(load_tab, text='Save model', command=self.save_model)
        save_model_button.grid(row=2, column=1, pady=5, padx=5, sticky='news')
        load_tab.grid_columnconfigure(2, weight=1)

        show_graphs_button = ttk.Button(load_tab, text='Show graphs', command=self.show_graphs)
        show_graphs_button.grid(row=2, column=2, pady=5, padx=5, sticky='news')
        load_tab.grid_columnconfigure(1, weight=1)

        self.frame = ttk.LabelFrame(self.pwindow, text='Show weights and biases')
        self.pwindow.add(self.frame)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)

        self.frame.rowconfigure(1, weight=1)
        info_frame1 = ttk.Frame(self.frame)
        info_frame1.grid(row=0, column=0, sticky='news', pady=2)
        info_frame1.rowconfigure(0, weight=1)
        info_frame1.rowconfigure(1, weight=1)
        info_frame1.rowconfigure(2, weight=1)

        Label = ttk.Label(info_frame1, text="Choose weights for layer:")
        Label.grid(row=0, column=0,  sticky='news')

        weights = []
        self.Combobox_weights = ttk.Combobox(info_frame1, values=weights,  validate='key')
        self.Combobox_weights.grid(row=0, column=1, sticky='news')

        show_button = ttk.Button(info_frame1, text='Show info', command=self.show_model_w)
        show_button.grid(row=0, column=2, sticky='news')

        info_frame2 = ttk.Frame(self.frame)
        info_frame2.grid(row=0, column=1, sticky='news')

        info_frame2.rowconfigure(0, weight=1)
        info_frame2.rowconfigure(1, weight=1)
        info_frame2.rowconfigure(2, weight=1)

        Label = ttk.Label(info_frame2, text="Choose biases for layer:")
        Label.grid(row=0, column=0, sticky='new')

        beises = []
        self.Combobox_beises = ttk.Combobox(info_frame2, values=beises,  validate='key')
        self.Combobox_beises.grid(row=0, column=1, sticky='news')

        show_button = ttk.Button(info_frame2, text='Show info', command=self.show_model_b)
        show_button.grid(row=0, column=2, sticky='news')


        frame = ScrollableFrame(self.frame)
        frame.grid(row=1, column=0, sticky='news')
        self.dataset_viewer = tkn.Text(frame.scrollable_frame)
        self.dataset_viewer.pack(fill=tkn.BOTH)

        frame = ScrollableFrame(self.frame)
        frame.grid(row=1, column=1, sticky='news')
        self.dataset_viewer2 = tkn.Text(frame.scrollable_frame)
        self.dataset_viewer2.pack(fill=tkn.BOTH)



    def show_model_w(self):
        layer = self.Combobox_weights.get()
        if layer is not None:
            weights, biases = self.final_modal.layers[int(layer)].get_weights()
            self.dataset_viewer.delete('1.0', tkn.END)
            self.dataset_viewer.insert(1.0, weights)

    def show_model_b(self):
        layer = self.Combobox_beises.get()
        if layer is not None:
            weights, biases = self.final_modal.layers[int(layer)].get_weights()
            self.dataset_viewer2.delete('1.0', tkn.END)
            self.dataset_viewer2.insert(1.0, biases)

    def save_model(self):
        model_name = self.network_name_entry.get()
        if self.final_modal is not None:
            self.final_modal.save(f"{model_name}")
            with open(f"{model_name}/history", 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)


    def show_graphs(self):
        # todo: пофиксить родителя, иначе происходит несанкционированное прилипание
        matplotlib.use('TkAgg')
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        fig.set_size_inches(10.5, 7.5)
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        plot_widget = canvas.get_tk_widget()

        plt.title("Losses train/validation")
        axs[0].plot(self.history.history["loss"], label="Train")
        axs[0].plot(self.history.history["val_loss"], label="Validation")
        plot_widget.grid(row=0, column=0, pady=5, padx=5)

        plt.title("Accuracies train/validation")
        axs[1].plot(self.history.history["accuracy"], label="Train")
        axs[1].plot(self.history.history["val_accuracy"], label="Validation")
        plot_widget.grid(row=1, column=0, pady=5, padx=5)
        plt.legend()
        plt.show()


    def create_layers(self):
        mas = []
        self.my_model = None
        self.final_modal = None
        self.layers.clear()
        layers = self.number_layers_entry.get()
        if layers.isdigit():
            layers = int(layers)
            for i in range(0, layers):
                mas.append(str(i))
            self.Combobox_pfl['values'] = mas
            for i in range(layers-1):
                self.layers.append([3, "relu"])
            self.layers.append([1, "sigmoid"])
            print(self.layers)

    def change_layer(self):
        layer_index = self.Combobox_pfl.get()
        neuron_numbers = self.number_neurons_entry.get()
        transfer_func = self.Combobox_transf.get()
        if layer_index is not None and neuron_numbers is not None and transfer_func is not None:
            self.layers[int(layer_index)] = [int(neuron_numbers), transfer_func]
            print(self.layers)

    def create_model(self):
        network_types = self.Combobox_nt.get()
        if network_types == "Sequential":
            network_types = tf.keras.Sequential()
        layers = self.layers
        loss = self.Combobox_lf.get()
        optimizer = self.Combobox_la.get()
        metrics = self.Combobox_mf.get()
        # print(network_types, layers, loss, optimizer, metrics)
        self.class_model = My_Model(network_types=network_types, layers_=layers, loss=loss, optimizer=optimizer,
                                  metrics=metrics)
        self.my_model = self.class_model.return_model()
        print(self.my_model)

    def load_pickle(self):
        self.input_frame = pd.read_pickle("input.pickle")
        self.output_frame = pd.read_pickle("output.pickle")
        self.stk_count = len(self.output_frame.index)
        self.const_count = self.stk_count
        self.current_table.updateModel(model=TableModel(dataframe=pd.DataFrame([['Training', 100, self.stk_count], ['Test', 0, self.stk_count_test]],
                     columns=['Dataset', 'Size %', 'Size str'])))
        self.current_table.redraw()

    def count_train(self):
        self.percent = self.procent_entry.get()
        if self.percent.isdigit():
            self.percent = int(self.percent)
            if 0 <= self.percent <= 100:
                self.stk_count_test = int(self.const_count * (self.percent) / 100)
                self.stk_count = self.const_count - self.stk_count_test
                self.current_table.updateModel(model=TableModel(dataframe=
                                                                pd.DataFrame(
                                                                    [['Training', 100 - self.percent, self.stk_count],
                                                                     ['Test', self.percent, self.stk_count_test]],
                                                                    columns=['Dataset', 'Size %', 'Size str'])))
                self.current_table.redraw()
                supervised = make_supervised(self.input_frame, self.output_frame)
                encoders = {}
                encoders = make_encoders(self.input_frame, encoders)
                encoders = make_encoders(self.output_frame, encoders)
                encoded_inputs = np.array(encode(supervised["inputs"], encoders))
                encoded_outputs = np.array(encode(supervised["outputs"], encoders))

                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                    encoded_inputs,
                    encoded_outputs,
                    test_size=self.percent / 100,
                    random_state=42)

    def learn_model(self):
        model = self.my_model
        epochs = self.number_epochs_entry.get()
        validation_split = self.validation_split_entry.get()
        if epochs is not None and validation_split is not None:
            epochs = int(epochs)
            validation_split = float(validation_split)
            self.history = model.fit(x=self.x_train, y=self.y_train, epochs=epochs, validation_split=validation_split)
            self.final_modal = model
            mas_i1 = []
            mas_i2 = []
            # todo: получить напрямую из модели, а не класса, кол-во слоев
            for i in range(self.class_model.return_layers_count()):
                mas_i1.append(i)
                mas_i2.append(i)
            self.Combobox_weights["values"] = mas_i1
            self.Combobox_beises["values"] = mas_i2


class My_Model():
    def __init__(self, network_types=None, layers_=None, loss=None, optimizer=None, metrics=None):
        self.model = network_types
        self.layers = len(layers_)
        for elem in layers_:
            self.model.add(tf.keras.layers.Dense(units=elem[0], activation=elem[1]))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    def return_model(self):
        return self.model

    def return_layers_count(self):
        return self.layers


def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = df[column].values
        result[column] = values
    return result


def make_supervised(raw_input_data, raw_output_data):
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data)}

def make_encoders(df, encoders):
    for column in df.columns:
        encoders[column] = lambda var: [var]
    return encoders

def encode(data, encoders):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")