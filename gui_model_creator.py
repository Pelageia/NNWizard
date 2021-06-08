import tkinter.ttk as ttk
import tkinter as tkn
from pandastable import Table, TableModel
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


class ModelCreator(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        self.pwindow = tkn.PanedWindow(self, orient="vertical")
        self.pwindow.pack(fill="both", expand=True)

        ## Панель вкладок
        top_part = ttk.Notebook(self.pwindow)
        self.pwindow.add(top_part)

        # --- Загрузка
        load_tab = ttk.Frame(top_part)
        top_part.add(load_tab, text='Разбиение данных', sticky='new')
        load_tab.grid_columnconfigure(0, weight=1)
        load_tab.grid_rowconfigure(0, weight=1)

        load_button = ttk.Button(load_tab, text='load pickles', command=self.load_pickle)
        load_button.grid(row=0, column=0, sticky='news')
        load_tab.grid_columnconfigure(0, weight=1)

        Label = ttk.Label(load_tab, text="Choose column for filtration")
        Label.grid(row=0, column=1, sticky='new')
        load_tab.grid_columnconfigure(1, weight=1)

        self.procent_entry = ttk.Entry(load_tab)
        self.procent_entry.grid(row=0, column=2, sticky='news')
        load_tab.grid_columnconfigure(2, weight=1)

        load_button = ttk.Button(load_tab, text='Разделение', command=self.count_train)
        load_button.grid(row=0, column=3, sticky='news')
        load_tab.grid_columnconfigure(3, weight=1)

        clean_tab = ttk.Frame(top_part)
        top_part.add(clean_tab, text='Разбиение данных', sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)
        clean_tab.grid_rowconfigure(0, weight=1)

        clear_button = ttk.Button(clean_tab, text='Разделение данных')
        clear_button.grid(row=0, column=0, sticky='new')
        clean_tab.grid_columnconfigure(0, weight=1)

        ## Информация о датасете
        bottom_part = ttk.Frame(self.pwindow)
        bottom_part.pack(fill='both')
        self.pwindow.add(bottom_part, stretch="always")
        ################################################
        # todo: сделать отдельно от инициализации класса ( кнопкой ) весь блок от считывания файла и отедльно функционал для обновления таблицы
        # self.input_frame = pd.read_pickle("input.pickle")
        # self.output_frame = pd.read_pickle("output.pickle")
        # self.stk_count = len(self.output_frame.index)
        self.stk_count = 0
        self.stk_count_test = 0
        df = None
        self.current_table = Table(bottom_part, dataframe=
        pd.DataFrame([['Training', 100, self.stk_count], ['Test', 0, self.stk_count_test]],
                     columns=['Dataset', 'Size %', 'Size str']),
                                   showtoolbar=0, showstatusbar=1)
        self.current_table.enable_menus = False
        self.current_table.grid(row=0, column=0, sticky='nsew')
        self.current_table.show()
        self.current_table.redraw()

    def load_pickle(self):
        self.input_frame = pd.read_pickle("input.pickle")
        self.output_frame = pd.read_pickle("output.pickle")
        self.stk_count = len(self.output_frame.index)
        self.current_table.updateModel(model=TableModel(dataframe=pd.DataFrame([['Training', 100, self.stk_count], ['Test', 0, self.stk_count_test]],
                     columns=['Dataset', 'Size %', 'Size str'])))
        self.current_table.redraw()
    def count_train(self):
        percent = self.procent_entry.get()
        if percent.isdigit():
            percent = int(percent)
            if 0 <= percent <= 100:
                self.stk_count_test = int(self.stk_count * (percent) / 100)
                self.stk_count = self.stk_count - self.stk_count_test
                self.current_table.updateModel(model=TableModel(dataframe=
                                                                pd.DataFrame(
                                                                    [['Training', 100 - percent, self.stk_count],
                                                                     ['Test', percent, self.stk_count_test]],
                                                                    columns=['Dataset', 'Size %', 'Size str'])))
                self.current_table.redraw()
                # inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(
                #     self.input_frame,
                #     self.output_frame,
                #     test_size=percent / 100,
                #     random_state=42)
                # print(inputs_train, inputs_test, expected_output_train, expected_output_test)
                supervised = make_supervised(self.input_frame, self.output_frame)
                encoders = {}
                encoders = make_encoders(self.input_frame, encoders)
                encoders = make_encoders(self.output_frame, encoders)
                encoded_inputs = encode(supervised["inputs"], encoders)
                encoded_outputs = encode(supervised["outputs"], encoders)
                print(encoded_inputs)
                print(encoded_outputs)
                # x_train, x_test, y_train, y_test = train_test_split(
                #     self.input_frame,
                #     self.output_frame,
                #     test_size=percent / 100,
                #     random_state=42)

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
