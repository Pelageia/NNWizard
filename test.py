##IMPORTS--------------------------------------------
from tkinter import *
import tkinter.font
##GUIWindow------------------------------------------
window = tkinter.Tk()
screen_w = window.winfo_screenwidth()
screen_h = window.winfo_screenheight()
window.geometry('%dx%d' % (screen_w, screen_h))
window.title("betaV00")
myFont = tkinter.font.Font(family = 'Helvetica', size = 12, weight = "bold")
##WIDGET_BottomButtons---------------------------------
h=6
bot_frame = Frame(window)
bot_frame.pack(side=BOTTOM, fill="x") #here

btn_HOME = Button(bot_frame, text="Home", font=myFont, bg='green', fg='white', height=h)
btn_HOME.grid(row = 0, column = 0, sticky="NESW") #here
btn_LEDS = Button(bot_frame, text="LEDS", font=myFont, bg='black', fg='green', height=h)
btn_LEDS.grid(row = 0, column = 1, sticky="NESW") #and here

bot_frame.columnconfigure(0, weight=1)
bot_frame.columnconfigure(1, weight=1)
##-----------------------------------------------------

window.mainloop()