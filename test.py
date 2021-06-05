from tkinter import *

root = Tk()
Grid.rowconfigure(root, 0, weight=1)
Grid.columnconfigure(root, 0, weight=1)

frame=Frame(root)
frame.grid(row=0, column=0, sticky=N+S+E+W)

grid=Frame(frame)
grid.grid(sticky=N+S+E+W, column=0, row=1, columnspan=3)
Grid.rowconfigure(frame, 0, weight=1)
Grid.columnconfigure(frame, 0, weight=1)

#example values
btn = Button(frame)
btn.grid(column=0, row=0, sticky=N+S+E+W)

root.mainloop()