#!/usr/bin/python3
#-*- coding: utf-8 -*-

'''
Main GUI structure
'''

import tkinter as tkn
import gui_elements as gel
from gui_data_manager import DataManager
#todo: импорт интерфейсов для отдельных менеджеров

'''
https://stackoverflow.com/questions/26097811/image-pyimage2-doesnt-exist
'''

class GUI():

    def __init__(self, main_window, data, *args, **kwargs):
        self.main_window = main_window
        self.data = data
        self.show_main_frame()
        self.show_toolbar()
        self.show_tools_frame()

    def show_main_frame(self):
        '''
        Main frame
        '''
        self.main_frame = tkn.Frame(self.main_window, bg='red')
        self.main_frame.pack(fill='both', expand=True)

    def show_toolbar(self):
        '''
        Toolbar
        '''
        toolbar = tkn.Frame(self.main_frame, bd=1, relief=tkn.RAISED, bg='green' )
        toolbar.pack(side=tkn.TOP, fill=tkn.X)

        # buttons launch modules
        #todo: добавить на кнопки картинки
        button_data = tkn.Button(toolbar, text='DataManager', command=lambda: self.show_tool(DataManager))
        button_data.grid(row=0, column=0)

        button_model = tkn.Button(toolbar, text='ModelCreator', command=lambda: self.show_tool(ModelCreator))
        button_model.grid(row=0, column=1)

        button_vis = tkn.Button(toolbar, text='Visualisation', command=lambda: self.show_tool(Visualisation))
        button_vis.grid(row=0, column=2)

        #todo: для каждого менеджера своя кнопка

    def show_tools_frame(self):
        '''
        Frame for each tool
        '''
        self.tools_frame = tkn.Frame(self.main_frame, bd=1, relief=tkn.RAISED)
        self.tools_frame.pack(side=tkn.TOP, fill=tkn.X)

        self.tools_frames = {}
        for F in (DataManager, ModelCreator, Visualisation): #todo: добавить ссылки на классы всех менеджеров
            frame = F(self.tools_frame, self)
            self.tools_frames[F] = frame
            frame.grid(row=0, column=0, sticky=tkn.NSEW)

        # указываем страницу, загружаемую по умолчанию
        self.show_tool(DataManager)

    def show_tool(self, tool_class_name):
        '''
        Show the frame of current tool
        '''
        self.current_frame = self.tools_frames[tool_class_name]
        self.current_frame.tkraise()

#todo: вынести в отдельные файлы
class ModelCreator(tkn.Frame):

    def __init__(self, parent, controller):
        tkn.Frame.__init__(self, parent)
        self.controller = controller
        l = tkn.Label(self,text = 'Hello ModelCreator!')
        l.pack()

class Visualisation(tkn.Frame):

    def __init__(self, parent, controller):
        tkn.Frame.__init__(self, parent)
        self.controller = controller
        l = tkn.Label(self,text = 'Hello Visualisation!')
        l.pack()