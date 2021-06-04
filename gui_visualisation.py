import tkinter as tkn

class Visualisation(tkn.Frame):
    def __init__(self, parent, controller):
        tkn.Frame.__init__(self, parent)
        self.controller = controller
        l = tkn.Label(self,text = 'Hello Visualisation!')
        l.pack()