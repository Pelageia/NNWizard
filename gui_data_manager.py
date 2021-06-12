import tkinter as tkn
from tkinter import LEFT
from tkinter import Canvas
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
from pandastable import Table, TableModel
from sklearn import preprocessing
import sklearn
from sklearn.model_selection import train_test_split
import tkinter.messagebox as mb

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


temp_df = None
input_data = None
output_data = None


class DataManager(ttk.Frame):

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
        top_part.add(load_tab, text='Загрузка', sticky='news')
        load_tab.grid_columnconfigure(0, weight=1)
        load_tab.grid_rowconfigure(0, weight=1)

        load_button = ttk.Button(load_tab, text='load set', command=self.load_set)
        load_button.grid(row=0, column=0, sticky='news')
        load_tab.grid_columnconfigure(0, weight=1)

        clear_button = ttk.Button(load_tab, text='clear set', command=self.clear_set)
        clear_button.grid(row=0, column=1, sticky='news')
        load_tab.grid_columnconfigure(1, weight=1)

        save_button = ttk.Button(load_tab, text='save set', command=self.save_set)
        save_button.grid(row=0, column=2, sticky='news')
        load_tab.grid_columnconfigure(2, weight=1)

        # --- Очистка
        clean_tab = ttk.Frame(top_part)
        top_part.add(clean_tab, text='Обработка данных', sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)
        clean_tab.grid_rowconfigure(0, weight=1)

        clear_button2 = ttk.Button(clean_tab, text='Фильтрация', command=self.filter)
        clear_button2.grid(row=0, column=0, sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)

        clear_button = ttk.Button(clean_tab, text='Визуализация', command=self.visualization)
        clear_button.grid(row=0, column=1, sticky='news')
        clean_tab.grid_columnconfigure(1, weight=1)

        clean_tab = ttk.Frame(top_part)
        top_part.add(clean_tab, text='Разбиение данных', sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)
        clean_tab.grid_rowconfigure(0, weight=1)

        clear_button = ttk.Button(clean_tab, text='Разделение данных', command=self.separate_data)
        clear_button.grid(row=0, column=0, sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)

        clear_button2 = ttk.Button(clean_tab, text='Выгрузить input and output', command=self.export_data)
        clear_button2.grid(row=0, column=1, sticky='news')
        clean_tab.grid_columnconfigure(1, weight=1)

        ## Информация о датасете
        bottom_part = ttk.Frame(self.pwindow)
        bottom_part.pack(fill='both')
        self.pwindow.add(bottom_part, stretch="always")

        ################################################
        df = None
        self.current_table = Table(bottom_part, dataframe=df, showtoolbar=0, showstatusbar=1)
        self.current_table.enable_menus = False
        self.current_table.grid(row=0, column=0, sticky='nsew')
        self.current_table.show()

    def clean_data(self):
        #### Вызов очистки датафрейма методом cleanData
        self.current_table.cleanData()

    def export_data(self):
        # global input_data, output_data
        input = self.current_table.model.df[input_data]
        # print(input.dtypes)

        catch_object = [col for col, dt in input.dtypes.items() if dt == object]
        if len(catch_object) > 0:
            mb.showerror("Error", "Denied try to export object data in input dataset")
        else:

            expected = self.current_table.model.df[output_data]
            catch_object = [col for col, dt in expected.dtypes.items() if dt == object]
            if len(catch_object) > 0:
                mb.showerror("Error", "Denied try to export object data in export dataset")
            else:
                input.to_pickle("input.pickle")
                expected.to_pickle("output.pickle")

    def print_info(self):
        #### Получаем датафрейм таблицы, если указать Имя колонки, то соотвествтенно содержимое конкретной колонки
        print(self.current_table.model.df['Name'])

    def apply_clean_set(self):
        self.clear_set_window.destroy()
        self.update_data_set()

    def drop_lines_set(self):
        # удалили строки
        self.local_dataset.dropna(inplace=True)
        # обновили вид локальной таблицы
        data = self.local_dataset.head()
        table = self.clear_set_window.local_dataset_viewer
        self.update_local_table(table, data)
        # обновим информацию для очистки
        data = self.local_dataset.isnull().sum()
        table = self.clear_set_window.info_viewer
        self.update_local_table(table, data)

    def separate_data(self):
        global temp_df, input_data, output_data
        SeparateDialog(self, table=self.current_table, input_data=input_data, output_data=output_data)
        if temp_df is not None and input_data is not None and output_data is not None:
            self.current_table.updateModel(model=TableModel(dataframe=temp_df))
            self.current_table.model.df
            for col in self.current_table.model.df.columns:
                if col in input_data:
                    self.current_table.columncolors[col] = '#54e600'
                else:
                    self.current_table.columncolors[col] = 'white'
            self.current_table.columncolors[output_data[0]] = '#990061'
            self.current_table.redraw()
            temp_df = None

    def filter(self):
        table = self.current_table
        FilterDialog(self, table=table)
        global temp_df
        if temp_df is not None:
            self.current_table.updateModel(model=TableModel(dataframe=temp_df))
            self.current_table.redraw()
            temp_df = None


    def save_set(self):
        self.current_table.save()

    def clear_set(self):
        self.current_table.clearTable()

    def load_set(self):
        table = self.current_table
        table.importCSV(dialog=True)

    def visualization(self):
        VisualizeDialog(self, self.current_table.model.df)

class FilterDialog(ttk.Frame):
    def __init__(self, parent=None, table=None):
        self.parent = parent
        self.table = table
        self.main = tkn.Toplevel()

        self.main.title('Data filtering')
        self.main.protocol("WM_DELETE_WINDOW", self.quit)
        self.main.grab_set()
        self.main.transient(parent)

        bf = tkn.Frame(self.main)
        bf.pack(side=LEFT, fill=tkn.BOTH)

        self.m = tkn.PanedWindow(self.main, orient=tkn.VERTICAL)
        self.m.pack(fill="both", expand=True)
        # self.m.pack(side=LEFT, fill=tkn.BOTH, expand=1)

        tf = tkn.Frame(self.main)
        self.m.add(tf)
        self.previewtable = Table(parent=tf, model=self.table.model, df=self.table.model.df, showstatusbar=1,
                                  showtoolbar=0, width=800, height=600)
        self.previewtable.enable_menus = False
        self.previewtable.show()

        optsframe = tkn.Frame(bf)
        optsframe.pack(side=tkn.TOP, fill=tkn.BOTH)

        columnNames = []
        for col in self.previewtable.model.df.columns:
            columnNames.append(col)

        self.Label = ttk.Label(bf, text="Choose column for filtration")
        self.Label.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        self.Combobox = ttk.Combobox(bf, values=columnNames, width=40, validate='key')
        self.Combobox.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)
        self.Combobox.current(0)
        b = tkn.Button(bf, text="average filtration", width=40, command=self.set_average_to_nans)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="delete line with emptiness", width=40, command=self.del_rows_with_emptiness)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="convert categorial to numbers", width=40, command=self.convert_categorial_to_nums)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="delete constants", width=40, command=self.delete_constants)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="delete duplicate rows", width=40, command=self.delete_duplicate_rows)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="delete duplicate cells", width=40, command=self.delete_duplicate_cells)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="normalize column", width=40, command=self.normalize_column)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="delete column", width=40, command=self.delete_column)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="Import", width=40, command=self.doImport)
        b.pack(side=tkn.BOTTOM, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="Cancel", width=40, command=self.quit)
        b.pack(side=tkn.BOTTOM, fill=tkn.BOTH, pady=2)
        self.main.wait_window()


    def quit(self):
        self.main.destroy()

    def normalize_column(self):
        colname = self.Combobox.get()
        if colname:
            df = self.previewtable.model.df
            x = df[[colname]].values.astype(float)
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df[colname] = x_scaled
            self.previewtable.updateModel(model=TableModel(dataframe=df))
            self.previewtable.redraw()

    def delete_column(self):
        colname = self.Combobox.get()
        if colname:
            df = self.previewtable.model.df
            df.drop([colname], axis=1, inplace=True)
            self.previewtable.model.df = df
            self.previewtable.updateModel(model=TableModel(dataframe=df))
            self.previewtable.redraw()

            columnNames = []
            for col in self.previewtable.model.df.columns:
                columnNames.append(col)
            self.Combobox['values'] = columnNames
            self.Combobox.current(0)

    def del_rows_with_emptiness(self):
        colname = self.Combobox.get()
        if colname:
            df = self.previewtable.model.df
            nan_values = df[
                df[colname].isna()]  # возвращает массив int64, но что бы удалить по индексу строки, нужен Int
            # хитрая махинация по удалению через преобразованный массив индексов в инт
            df.drop(df.index[pd.to_numeric(nan_values.index, downcast='signed')], inplace=True)
            self.previewtable.model.df = df
            self.previewtable.updateModel(model=TableModel(dataframe=df))
            self.previewtable.redraw()

    def set_average_to_nans(self):
        colname = self.Combobox.get()
        if colname:
            df = self.previewtable.model.df
            mean_value = df[colname].mean()
            df[colname] = df[colname].fillna(mean_value)
            self.previewtable.updateModel(model=TableModel(dataframe=df))
            self.previewtable.redraw()

    def convert_categorial_to_nums(self):
        colname = self.Combobox.get()
        if colname:
            df = self.previewtable.model.df
            df[colname] = pd.Categorical(df[colname]).codes  # 0, 1, 2, 3
            # ???? перевод в 5 -> [0,0,0,0,1] или в elem/MAX
            self.previewtable.updateModel(model=TableModel(dataframe=df))
            self.previewtable.redraw()

    def delete_constants(self):
        df = self.previewtable.model.df
        df = df.loc[:, (df != df.iloc[0]).any()]
        self.previewtable.updateModel(model=TableModel(dataframe=df))
        self.previewtable.redraw()

    def delete_duplicate_rows(self):
        df = self.previewtable.model.df.drop_duplicates()
        self.previewtable.updateModel(model=TableModel(dataframe=df))
        self.previewtable.redraw()

    def delete_duplicate_cells(self):
        df = self.previewtable.model.df
        df = df.loc[:, ~df.columns.duplicated()]
        self.previewtable.updateModel(model=TableModel(dataframe=df))
        self.previewtable.redraw()

    def doImport(self):
        global temp_df
        temp_df = self.previewtable.model.df
        self.main.destroy()


class SeparateDialog(ttk.Frame):
    def __init__(self, parent=None, table=None, input_data=None, output_data=None):
        self.parent = parent
        self.table = table
        self.main = tkn.Toplevel()

        self.main.title('Data separating')
        self.main.protocol("WM_DELETE_WINDOW", self.quit)
        self.main.grab_set()
        self.main.transient(parent)

        bf = tkn.Frame(self.main)
        bf.pack(side=LEFT, fill=tkn.BOTH)

        self.m = tkn.PanedWindow(self.main, orient=tkn.VERTICAL)
        self.m.pack(fill="both", expand=True)

        tf = tkn.Frame(self.main)
        self.m.add(tf)
        self.previewtable = Table(parent=tf, model=self.table.model, df=self.table.model.df, showstatusbar=1,
                                  showtoolbar=0, width=800, height=600)
        self.previewtable.enable_menus = False
        self.previewtable.show()
        temp_df = self.previewtable.model.df
        if input_data is not None and output_data is not None:
            for col in input_data:
                self.previewtable.setColumnColors(clr='#54e600', cols=temp_df.columns.get_loc(col))
            self.previewtable.setColumnColors(clr='#990061', cols=temp_df.columns.get_loc(output_data[0]))


        optsframe = tkn.Frame(bf)
        optsframe.pack(side=tkn.TOP, fill=tkn.BOTH)

        columnNames = []
        for col in self.previewtable.model.df.columns:
            columnNames.append(col)

        self.Label = ttk.Label(optsframe, text="Choose input data")
        self.Label.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        self.cols_vals = []
        frame = ScrollableFrame(bf)
        frame.pack(side=tkn.TOP)
        for col in self.previewtable.model.df.columns:
            cols_vals_checkbox = ttk.Checkbutton(frame.scrollable_frame, text=col,
                                                 onvalue=1, offvalue=0)

            cols_vals_checkbox.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)
            self.cols_vals.append(cols_vals_checkbox)

        self.Label_out = ttk.Label(bf, text="Choose output data")
        self.Label_out.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        self.Combobox = ttk.Combobox(bf, values=columnNames, width=40, validate='key')
        self.Combobox.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="confirm changes", width=40, command=self.del_unselected_columns)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="Import", width=40, command=self.doImport)
        b.pack(side=tkn.BOTTOM, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="Cancel", width=40, command=self.quit)
        b.pack(side=tkn.BOTTOM, fill=tkn.BOTH, pady=2)
        self.input_data = None
        self.output_data = None
        self.main.wait_window()
        return

    def quit(self):
        self.main.destroy()
        return

    def del_unselected_columns(self):
        colname = self.Combobox.get()
        # rows_to_delete = []
        df = self.previewtable.model.df
        rows_to_allow = []
        for checkbox in self.cols_vals:
            if 'selected' in checkbox.state():
                rows_to_allow.append(checkbox.cget("text"))
        flag = True
        for col in rows_to_allow:
            if col == colname:
                flag = False
                mb.showerror("Error", "Chosen the same target as input")
        if flag:
            for col in self.previewtable.model.df.columns:
                if col in rows_to_allow:
                    self.previewtable.columncolors[col] = '#54e600'
                else:
                    self.previewtable.columncolors[col] = 'white'
            self.input_data = rows_to_allow
            if colname:
                self.previewtable.columncolors[colname] = '#990061'

            self.previewtable.redraw()
            self.output_data = [colname]

    def doImport(self):
        global temp_df, input_data, output_data
        temp_df = self.previewtable.model.df
        input_data = self.input_data
        output_data = self.output_data
        self.previewtable = None
        self.main.destroy()
        return


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


class VisualizeDialog(ttk.Frame):
    def __init__(self, parent=None, dataframe=None):
        self.parent = parent
        self.dataframe = dataframe
        self.main = tkn.Toplevel()

        self.main.title('Data visualization')
        self.main.protocol("WM_DELETE_WINDOW", self.quit)
        self.main.grab_set()
        self.main.transient(parent)

        bf = tkn.Frame(self.main)
        bf.pack(fill=tkn.BOTH)

        load_tab = ttk.LabelFrame(bf, text='Choose first variable')
        load_tab.pack(fill=tkn.BOTH)

        columnNames = []

        catch_object = [col for col, dt in self.dataframe.dtypes.items() if dt == object]

        for col in self.dataframe.columns:
            if col not in catch_object:
                columnNames.append(col)

        self.Combobox_n1 = ttk.Combobox(load_tab, values=columnNames, width=40, validate='key')
        self.Combobox_n1.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        load_tab = ttk.LabelFrame(bf, text='Choose procents to visualize')
        load_tab.pack(fill=tkn.BOTH)

        procents = ["100", "75", "50", "25"]
        self.Combobox_n2 = ttk.Combobox(load_tab, values=procents, width=40, validate='key')
        self.Combobox_n2.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="Visualize", width=40, command=self.make_visualization)
        b.pack(side=tkn.TOP, fill=tkn.BOTH, pady=2)

        b = tkn.Button(bf, text="Close", width=40, command=self.quit)
        b.pack(side=tkn.BOTTOM, fill=tkn.BOTH, pady=2)
        self.input_data = None
        self.output_data = None

        self.graph_frame = ttk.Frame(bf)

        self.main.wait_window()
        return

    def quit(self):
        self.main.destroy()
        return

    def make_visualization(self):
        colname_1 = self.Combobox_n1.get()
        procent = int(self.Combobox_n2.get())
        df = self.dataframe
        kolvo = self.dataframe.shape[0]
        matplotlib.use('TkAgg')
        fig = plt.figure(1)
        fig.set_size_inches(10.5, 7.5)
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        plot_widget = canvas.get_tk_widget()

        plt.title(f"{colname_1}")
        plt.stem(self.dataframe[colname_1][:int(kolvo*procent/100)], label=f"{colname_1}")
        plot_widget.grid(row=0, column=0, pady=5, padx=5)

        plt.legend()
        plt.show()