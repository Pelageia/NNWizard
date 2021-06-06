import tkinter as tkn
from collections import OrderedDict
from tkinter import filedialog as fd
from tkinter import Checkbutton
from tkinter import LEFT
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText

from pandastable import Table, TableModel



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

        load_button = tkn.Button(load_tab, text='load set', command=self.load_set)
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
        top_part.add(clean_tab, text ='Обработка данных', sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)
        clean_tab.grid_rowconfigure(0, weight=1)

        clear_button = ttk.Button(clean_tab, text='Нормализация', command=self.clean_data)
        clear_button.grid(row=0, column=0, sticky='news')
        clean_tab.grid_columnconfigure(0, weight=1)

        clear_button2 = ttk.Button(clean_tab, text='Фильтрация', command=self.filter)
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

        # # --- Сохранение
        # save_tab = ttk.Frame(top_part)
        # top_part.add(save_tab, text = 'Сохранение', sticky='news')
        # save_tab.grid_rowconfigure(0, weight=1)
        # save_tab.grid_columnconfigure(0, weight=1)



        ## Информация о датасете
        bottom_part = ttk.Frame(self.pwindow)
        bottom_part.pack(fill='both')
        self.pwindow.add(bottom_part, stretch="always")

################################################
        df = None
        # f1 = tkn.Frame(self)
        #
        self.current_table = Table(bottom_part, dataframe=df, showtoolbar=0, showstatusbar=1)
        self.current_table.grid(row=0, column=0, sticky='nsew')
        self.current_table.show()
        #self.dataset_info.config(state=ttk.DISABLED)
        # scy = ttk.Scrollbar(bottom_part,command=self.dataset_viewer.yview)
        # self.dataset_viewer.configure(yscrollcommand=scy.set)
        # self.dataset_viewer.grid(row=0, column=0, sticky='news')
        # scy.grid(row=0, column=1, sticky='ns')

    def clean_data(self):
        #### Вызов очистки датафрейма методом cleanData
        self.current_table.cleanData()

    def print_info(self):
        #### Получаем датафрейм таблицы, если указать Имя колонки, то соотвествтенно содержимое конкретной колонки
        print(self.current_table.model.df['Name'])
        print(self.current_table.model.getColumnType(1))
        print(self.current_table.model.getColumnType(3))
        print(self.current_table.model.getColumnType(8))


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


    def transf_set(self):
        pass


    def filter(self):
        FilterDialog(self, table=self.current_table)

    def save_set(self):
        self.current_table.save()

    def clear_set(self):
        self.current_table.clearTable()

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

class FilterDialog(tkn.Frame):

    def __init__(self, parent=None, table=None):
        self.parent = parent
        self.table = table
        self.main = tkn.Toplevel()
        self.master = self.main
        self.main.title('Data filtering')
        self.main.protocol("WM_DELETE_WINDOW", self.quit)
        self.main.grab_set()
        self.main.transient(parent)

        bf = tkn.Frame(self.main)
        bf.pack(side=LEFT, fill=tkn.BOTH)

        self.m = tkn.PanedWindow(self.main, orient=tkn.VERTICAL)
        self.m.pack(side=LEFT, fill=tkn.BOTH, expand=1)

        tf = tkn.Frame(self.main)
        self.m.add(tf)
        self.previewtable = Table(parent=tf, model=self.table.model, df=self.table.model.df, showstatusbar=1, showtoolbar=0)

        self.previewtable.show()

        self.textpreview = ScrolledText(self.main, width=100, height=10, bg='white')
        self.m.add(self.textpreview)
        optsframe = tkn.Frame(bf)
        optsframe.pack(side=tkn.TOP,fill=tkn.BOTH)

        b = tkn.Button(bf, text="Update preview")
        b.pack(side=tkn.TOP,fill=tkn.X,pady=2)
        b = tkn.Button(bf, text="Import")
        b.pack(side=tkn.TOP,fill=tkn.X,pady=2)
        b = tkn.Button(bf, text="Cancel")
        b.pack(side=tkn.TOP,fill=tkn.X,pady=2)
        self.main.wait_window()
        return

    def quit(self):
        self.main.destroy()
        return

