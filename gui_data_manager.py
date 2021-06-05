import tkinter as tkn
from tkinter import filedialog as fd
from tkinter import Checkbutton
from tkinter import LEFT
import tkinter.ttk as ttk
from pandastable import Table


class DataManager(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        self.pwindow = tkn.PanedWindow(self, orient="vertical")
        self.pwindow.pack(fill="both", expand=True)

        ## Панель вкладок
        top_part = ttk.Notebook(self.pwindow)
        self.pwindow.add(top_part ,stretch="always")

        # --- Загрузка
        load_tab = ttk.Frame(top_part)
        top_part.add(load_tab, text ='Загрузка')
        load_tab.grid_rowconfigure(0, weight=1)
        load_tab.grid_columnconfigure(0, weight=1)

        load_button = ttk.Button(load_tab, text='load set', command=self.load_set)
        load_button.pack()

        # --- Очистка
        clean_tab = ttk.Frame(top_part)
        top_part.add(clean_tab, text ='Обработка данных')
        # clean_tab.grid_rowconfigure(0, weight=1)
        # clean_tab.grid_columnconfigure(0, weight=1)


        clear_button = ttk.Button(clean_tab, text='Нормализация', command=self.clear_set)
        clear_button.grid(row=0, column=0)
        clear_button2 = ttk.Button(clean_tab, text='Фильтрация', command=self.clear_set)
        clear_button2.grid(row=0, column=1)
        clear_button3 = ttk.Button(clean_tab, text='Перевод категориальных в количественные ', command=self.clear_set)
        clear_button3.grid(row=0, column=2)
        clear_button3 = ttk.Button(clean_tab, text='Удаление конст признаков', command=self.clear_set)
        clear_button3.grid(row=0, column=3)

        # --- Трансформация
        transf_tab = ttk.Frame(top_part)
        top_part.add(transf_tab, text ='Трансформация')
        transf_tab.grid_rowconfigure(0, weight=1)
        transf_tab.grid_columnconfigure(0, weight=1)

        transf_button = ttk.Button(transf_tab, text='transform set', command=self.transf_set)
        transf_button.pack()

        # --- Сохранение
        save_tab = ttk.Frame(top_part)
        top_part.add(save_tab, text ='Сохранение')
        save_tab.grid_rowconfigure(0, weight=1)
        save_tab.grid_columnconfigure(0, weight=1)

        save_button = ttk.Button(save_tab, text='save set', command=self.save_set)
        save_button.pack()

        ## Информация о датасете
        bottom_part = ttk.Frame(self.pwindow)
        bottom_part.pack(fill='both', expand=True)
        self.pwindow.add(bottom_part, stretch="always")


        # bottom_part.grid_rowconfigure(0, weight=1)
        # bottom_part.grid_columnconfigure(0, weight=1)

        # todo: заменить текст на таблицу
        # self.dataset_viewer = tkn.Text(bottom_part, width=50, height=20)
        # # self.dataset_info.config(state=ttk.DISABLED)
        # scy = ttk.Scrollbar(bottom_part, command=self.dataset_viewer.yview)
        # self.dataset_viewer.configure(yscrollcommand=scy.set)
        # self.dataset_viewer.grid(row=0, column=0, sticky='news')
        # scy.grid(row=0, column=1, sticky='ns')

################################################
        df = None
        # f1 = tkn.Frame(self)
        #
        self.current_table = Table(bottom_part, dataframe=df, showtoolbar=1, showstatusbar=1)
        self.current_table.grid(row=0, column=0, sticky='ew')
        self.current_table.show()
        #self.dataset_info.config(state=ttk.DISABLED)
        # scy = ttk.Scrollbar(bottom_part,command=self.dataset_viewer.yview)
        # self.dataset_viewer.configure(yscrollcommand=scy.set)
        # self.dataset_viewer.grid(row=0, column=0, sticky='news')
        # scy.grid(row=0, column=1, sticky='ns')

    def clear_set(self):
        # create local copy of a dataset
        self.local_dataset = self.controller.data.csv_data.copy()

        # create new window
        self.clear_set_window = tkn.Toplevel(self)

        # Button Frame
        button_frame = tkn.Frame(self.clear_set_window)
        button_frame.grid(row=0, column=0)

        drop_button = ttk.Button(button_frame, text='Drop', command=self.drop_lines_set)
        drop_button.grid(row=0, column=0)

        fill_button = ttk.Button(button_frame, text='Fill', command=self.fill_lines_set)
        fill_button.grid(row=0, column=1)

        apply_button = ttk.Button(button_frame, text='Apply', command=self.apply_clean_set)
        apply_button.grid(row=0, column=2)

        # Local Viewer Frame
        local_viewer_frame = tkn.Frame(self.clear_set_window)
        local_viewer_frame.grid(row=1, column=0)

        #todo: заменить на таблицу
        self.clear_set_window.local_dataset_viewer = tkn.Text(local_viewer_frame, width=50, height=20)
        scy = ttk.Scrollbar(local_viewer_frame,command=self.clear_set_window.local_dataset_viewer.yview)
        self.clear_set_window.local_dataset_viewer.configure(yscrollcommand=scy.set)
        self.clear_set_window.local_dataset_viewer.grid(row=0, column=0, sticky='news')
        scy.grid(row=0, column=1, sticky='ns')

        # Info Frame
        self.clear_set_window.info_frame = tkn.Frame(self.clear_set_window)
        self.clear_set_window.info_frame.grid(row=0, column=1, rowspan=2)

        #todo: заменить на таблицу
        self.clear_set_window.info_viewer = tkn.Text(self.clear_set_window.info_frame, width=50, height=20)
        scy = ttk.Scrollbar(local_viewer_frame,command=self.clear_set_window.info_viewer.yview)
        self.clear_set_window.info_viewer.configure(yscrollcommand=scy.set)
        self.clear_set_window.info_viewer.grid(row=0, column=0, sticky='news')
        scy.grid(row=0, column=1, sticky='ns')

        # выбрать столбцы участвующие в обучении
        pass

        # обновили вид локальной таблицы
        data = self.local_dataset.head()
        table = self.clear_set_window.local_dataset_viewer
        self.update_local_table(table,data)

        # обновим информацию для очистки
        data = self.local_dataset.isnull().sum()
        table = self.clear_set_window.info_viewer
        self.update_local_table(table, data)

    def update_data_set(self):
        '''
        обновлять данные в дата DataStorage
        '''
        pass

    def apply_clean_set(self):
        self.clear_set_window.destroy()
        self.update_data_set()

    def update_local_table(self, table, data):
        '''
        Обновлять данные внутри окошка
        '''
        table.delete('1.0', tkn.END)
        table.insert(1.0, data)

    def drop_lines_set(self):
        # удалили строки
        self.local_dataset.dropna(inplace=True)
        # обновили вид локальной таблицы
        data = self.local_dataset.head()
        table = self.clear_set_window.local_dataset_viewer
        self.update_local_table(table,data)
        # обновим информацию для очистки
        data = self.local_dataset.isnull().sum()
        table = self.clear_set_window.info_viewer
        self.update_local_table(table, data)

    def fill_lines_set(self):
        # удалили строки
        pass
        # обновили вид локальной таблицы
        self.update_local_table()

    def transf_set(self):
        pass

    def save_set(self):
        pass

    def show_load_set_window(self):
        '''
        '''
        self.load_set_window = tkn.Toplevel(self)

        load_button = ttk.Button(self.load_set_window, text='Load', command=self.load_set)
        load_button.grid(row=0, column=0)

        self.first_string=1

        self.cb = Checkbutton(
            self.load_set_window, text="Первая строка является заголовком", variable=self.first_string,
            onvalue=1, offvalue=0)
        self.cb.grid(row=1, column=0)
        close_button = ttk.Button(self.load_set_window, text='OK', command=self.OK_button_click)
        close_button.grid(row=2, column=0)

        frame_for_text = ttk.Frame(self.load_set_window)
        frame_for_text.grid(row=0, column=1, sticky='news')
        frame_for_text.grid_rowconfigure(0, weight=1)
        frame_for_text.grid_columnconfigure(0, weight=1)

        self.dataset_info = tkn.Text(frame_for_text, width=100, height=40)
        #self.dataset_info.config(state=ttk.DISABLED)
        scy = ttk.Scrollbar(frame_for_text,command=self.dataset_info.yview)
        self.dataset_info.configure(yscrollcommand=scy.set)
        self.dataset_info.grid(row=0, column=0, sticky='news')
        scy.grid(row=0, column=1, sticky='ns')



    def OK_button_click(self):
        self.load_set_window.destroy()
        self.dataset_viewer.delete('1.0', tkn.END)
        self.dataset_viewer.insert(1.0, self.controller.data.head_csv())

    def load_set(self):

        # file_name = fd.askopenfilename(filetypes=(("csv", "*.csv"), ("All files", "*.*")))
        # self.controller.data.file_path = file_name
        # self.controller.data.load_csv()
        #
        # #self.dataset_info.config(state=ttk.WRITABLE)
        # self.dataset_info.delete('1.0', tkn.END)
        # self.dataset_info.insert(1.0, self.controller.data.read_clean_csv())
        # #self.dataset_info.config(state=ttk.DISABLED)

        table = self.current_table
        table.importCSV(dialog=True)