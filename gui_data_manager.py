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
        self.pwindow.add(top_part, stretch="always")

        # --- Загрузка
        load_tab = ttk.Frame(top_part)
        top_part.add(load_tab, text ='Загрузка', sticky='news')
        load_tab.grid_columnconfigure(0, weight=1)
        load_tab.grid_rowconfigure(0, weight=1)

        load_button = ttk.Button(load_tab, text='load set', command=self.load_set)
        load_button.grid(row=0, column=0, sticky='news')

        # --- Очистка
        clean_tab = ttk.Frame(top_part)
        top_part.add(clean_tab, text ='Обработка данных', sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)
        clean_tab.grid_rowconfigure(0, weight=1)

        clear_button = ttk.Button(clean_tab, text='Нормализация', command=self.clear_set)
        clear_button.grid(row=0, column=0, sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)

        clear_button2 = ttk.Button(clean_tab, text='Фильтрация')
        clear_button2.grid(row=0, column=1, sticky='news')
        clean_tab.grid_columnconfigure(1, weight=1)

        clear_button3 = ttk.Button(clean_tab, text='Перевод категориальных в количественные ')
        clear_button3.grid(row=0, column=2, sticky='news')
        clean_tab.grid_columnconfigure(2, weight=1)

        clear_button3 = ttk.Button(clean_tab, text='Удаление конст признаков')
        clear_button3.grid(row=0, column=3, sticky='news')
        clean_tab.grid_columnconfigure(3, weight=1)

        clear_button4 = ttk.Button(clean_tab, text='Вывод информации в консоль', command=self.print_info)
        clear_button4.grid(row=0, column=4, sticky='news')
        clean_tab.grid_columnconfigure(4, weight=1)

        # --- Трансформация
        transf_tab = ttk.Frame(top_part)
        top_part.add(transf_tab, text ='Трансформация', sticky='news')
        transf_tab.grid_rowconfigure(0, weight=1)
        transf_tab.grid_columnconfigure(0, weight=1)

        transf_button = ttk.Button(transf_tab, text='transform set', command=self.transf_set)
        transf_button.grid(row=0, column=0, sticky='news')
        transf_button.grid_columnconfigure(0, weight=1)

        # --- Сохранение
        save_tab = ttk.Frame(top_part)
        top_part.add(save_tab, text = 'Сохранение', sticky='news')
        save_tab.grid_rowconfigure(0, weight=1)
        save_tab.grid_columnconfigure(0, weight=1)

        save_button = ttk.Button(save_tab, text='save set', command=self.save_set)
        save_button.grid(row=0, column=0, sticky='news')
        save_button.grid_columnconfigure(0, weight=1)

        ## Информация о датасете
        bottom_part = ttk.Frame(self.pwindow)
        bottom_part.pack(fill='both', expand=True)
        self.pwindow.add(bottom_part, stretch="always")

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
        #### Вызов очистки датафрейма методом cleanData
        self.current_table.cleanData()

    def print_info(self):
        #### Получаем датафрейм таблицы, если указать Имя колонки, то соотвествтенно содержимое конкретной колонки
        print(self.current_table.model.df['Name'])

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