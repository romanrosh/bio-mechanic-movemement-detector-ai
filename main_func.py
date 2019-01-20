from tkinter import *


def button1(param):
    print('inside button1', param)


def button2(param):
    print('inside button2', param)


def button3(param):
    print('inside button3', param)


def main():
    input_width = 150
    root = Tk()
    root.resizable(width=FALSE, height=FALSE)
    root.geometry("630x700")

    button1inputtext = Label(root, text='Template for data preparation resize, flip, rename pics', anchor=W)
    button1inputtext.place(x=10, y=10, height=25)

    button1inputtext = Label(root, text='Button 1', anchor=W)
    button1inputtext.place(x=10, y=70, width=250, height=25)
    button1input = Entry(root)
    button1input.place(x=10, y=100, width=70 + (input_width + 20) * 2, height=25)
    goButton1 = Button(root, text='button1', command=lambda: button1(button1input.get()))
    goButton1.config(relief=RAISED)
    goButton1.place(x=450, y=100, width=100, height=25)

    button2inputtext = Label(root, text='Button 2', anchor=W)
    button2inputtext.place(x=10, y=120, width=250, height=25)
    button2input = Entry(root)
    button2input.place(x=10, y=150, width=70 + (input_width + 20) * 2, height=25)
    goButton2 = Button(root, text='button2', command=lambda: button2(button2input.get()))
    goButton2.config(relief=RAISED)
    goButton2.place(x=450, y=150, width=100, height=25)

    button3inputtext = Label(root, text='Button 3', anchor=W)
    button3inputtext.place(x=10, y=170, width=250, height=25)
    button3input = Entry(root)
    button3input.place(x=10, y=200, width=70 + (input_width + 20) * 2, height=25)
    goButton3 = Button(root, text='button3', command=lambda: button3(button3input.get()))
    goButton3.config(relief=RAISED)
    goButton3.place(x=450, y=200, width=100, height=25)

    ExitButton = Button(root, text='Exit', command=exit)
    ExitButton.config(relief=RAISED)
    ExitButton.place(x=500, y=400, width=100, height=25)

    root.mainloop()


if __name__ == '__main__':
    main()
