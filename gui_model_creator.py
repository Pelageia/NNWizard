import tkinter.ttk as ttk
import tkinter as tkn
from pandastable import Table, TableModel
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
# import keras as k


class ModelCreator(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        ## Панель вкладок
        top_part = ttk.Notebook(self)
        top_part.pack(fill='both')

        first_frame = ttk.Frame(top_part)
        top_part.add(first_frame, text='Разбиение данных', sticky='new')

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
        self.number_neurons_entry.insert(0, "2")
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
        top_part.add(third_frame, text='Обучение модели', sticky='new')

        self.pwindow = tkn.PanedWindow(third_frame, orient="vertical")
        self.pwindow.pack(fill="both", expand=True)

        # # --- Загрузка
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


        # load_button = ttk.Button(load_tab, text='Learn model', command=self.learn_model)
        # load_button.grid(row=0, column=0, sticky='news')
        # load_tab.grid_columnconfigure(0, weight=1)

        load_button = ttk.Button(load_tab, text='Learn model', command=self.learn_model)
        load_button.grid(row=2, column=0, pady=5, padx=5, sticky='news')
        load_tab.grid_columnconfigure(2, weight=1)

        self.my_model = None



    def create_layers(self):
        mas = []
        self.layers.clear()
        layers = self.number_layers_entry.get()
        if layers.isdigit():
            layers = int(layers)
            for i in range(0, layers):
                mas.append(str(i))
            self.Combobox_pfl['values'] = mas
            for i in range(layers-1):
                #k.layers.Dense(units=3, activation="relu")
                #k.layers.Dense(units=1, activation="sigmoid")
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
        self.my_model = My_Model(network_types=network_types, layers_=layers, loss=loss, optimizer=optimizer,
                                  metrics=metrics).return_model()
        print(self.my_model)

    def load_pickle(self):
        self.input_frame = pd.read_pickle("input.pickle")
        self.output_frame = pd.read_pickle("output.pickle")
        self.stk_count = len(self.output_frame.index)
        self.current_table.updateModel(model=TableModel(dataframe=pd.DataFrame([['Training', 100, self.stk_count], ['Test', 0, self.stk_count_test]],
                     columns=['Dataset', 'Size %', 'Size str'])))
        self.current_table.redraw()

    def count_train(self):
        self.percent = self.procent_entry.get()
        if self.percent.isdigit():
            self.percent = int(self.percent)
            if 0 <= self.percent <= 100:
                self.stk_count_test = int(self.stk_count * (self.percent) / 100)
                self.stk_count = self.stk_count - self.stk_count_test
                self.current_table.updateModel(model=TableModel(dataframe=
                                                                pd.DataFrame(
                                                                    [['Training', 100 - self.percent, self.stk_count],
                                                                     ['Test', self.percent, self.stk_count_test]],
                                                                    columns=['Dataset', 'Size %', 'Size str'])))
                self.current_table.redraw()
                # inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(
                #     self.input_frame,
                #     self.output_frame,
                #     test_size=percent / 100,
                #     random_state=42)
                # print(inputs_train, inputs_test, expected_output_train, expected_output_test)

    def learn_model(self):
        supervised = make_supervised(self.input_frame, self.output_frame)
        encoders = {}
        encoders = make_encoders(self.input_frame, encoders)
        encoders = make_encoders(self.output_frame, encoders)
        encoded_inputs = np.array(encode(supervised["inputs"], encoders))
        encoded_outputs = np.array(encode(supervised["outputs"], encoders))

        x_train, x_test, y_train, y_test = train_test_split(
            encoded_inputs,
            encoded_outputs,
            test_size=self.percent / 100,
            random_state=42)
        model = self.my_model
        epochs = self.number_epochs_entry.get()
        validation_split = self.validation_split_entry.get()
        if epochs is not None and validation_split is not None:
            epochs = int(epochs)
            validation_split = float(validation_split)
            fit_results = model.fit(x=x_train, y=y_train, epochs=epochs, validation_split=validation_split)


class My_Model():
    def __init__(self, network_types=None, layers_=None, loss=None, optimizer=None, metrics=None):
        self.model = network_types
        for elem in layers_:
            self.model.add(tf.keras.layers.Dense(units=elem[0], activation=elem[1]))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    def return_model(self):
        return self.model






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