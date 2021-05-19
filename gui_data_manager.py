import tkinter as tkn
from tkinter import filedialog as fd
import tkinter.ttk as ttk

class DataManager(tkn.Frame):

    def __init__(self, parent, controller):
        tkn.Frame.__init__(self, parent, bg='yellow')
        self.controller = controller
        l = tkn.Label(self,text = 'Hello DataManager!')
        l.pack()

        load_button = tkn.Button(self, text='load set', command=self.show_load_set_window)
        load_button.pack()

    def show_load_set_window(self):
        '''
        '''
        cnw = tkn.Toplevel(self)

        load_button = tkn.Button(cnw, text='Load', command=self.load_set)
        load_button.grid(row=0, column=0)

        frame_for_text = tkn.Frame(cnw)
        frame_for_text.grid(row=1, column=0, sticky='news')
        frame_for_text.grid_rowconfigure(0, weight=1)
        frame_for_text.grid_columnconfigure(0, weight=1)

        self.dataset_info = tkn.Text(frame_for_text, width=50, height=20)
        #self.dataset_info.config(state=tkn.DISABLED)
        scy = ttk.Scrollbar(frame_for_text,command=self.dataset_info.yview)
        self.dataset_info.configure(yscrollcommand=scy.set)
        self.dataset_info.grid(row=0, column=0, sticky='news')
        scy.grid(row=0, column=1, sticky='ns')

        close_button = tkn.Button(cnw, text='OK', command=lambda: cnw.destroy())
        close_button.grid(row=2, column=0)

    def load_set(self):

        file_name = fd.askopenfilename(filetypes=(("csv", "*.csv"), ("All files", "*.*")))
        self.controller.data.file_path = file_name
        self.controller.data.load_csv()
        #self.dataset_info.config(state=tkn.WRITABLE)
        self.dataset_info.delete('1.0', tkn.END)
        self.dataset_info.insert(1.0, self.controller.data.get_csv_info())
        #self.dataset_info.config(state=tkn.DISABLED)


