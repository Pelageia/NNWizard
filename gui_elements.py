#!/usr/bin/python3
#-*- coding: utf-8 -*-

'''
Different elements for the GUI
'''

import tkinter as tkn

class AltWindow():
    '''
    Alt window
    '''

    def _create(self, event):
        self.window = tkn.Toplevel(bg='yellow')
        self.window.overrideredirect(1)
        label = tkn.Label(self.window, text=self.text, bg='yellow')
        label.pack(side=tkn.BOTTOM)
        self.window.wm_geometry('+{}+{}'.format(int(event.x_root)+5,int(event.y_root)+15))

    def _destroy(self, event):
        self.window.destroy()

def create_alt_window(obj, text):
    '''
    Show alt window
    '''
    popup_window = AltWindow()
    popup_window.text = text
    obj.bind('<Enter>',popup_window._create)
    obj.bind('<Leave>',popup_window._destroy)