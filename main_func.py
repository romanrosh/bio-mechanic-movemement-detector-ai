from tkinter import *
import cv2
import OpenPoseVideo

OUTPUT = './destination/'
MODE = 'BODY25'


def video_cut(source):
    """
    cutting video to pictures and saving to destination folder
    folder should exist
    :param source:
    :return:
    """

    cap = cv2.VideoCapture(source)

    counter = 0

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, dsize=(400, 400))
            cv2.imwrite(OUTPUT+'/' + str(counter) + '.jpg', gray)
            counter += 1
            cv2.imshow('frame', gray)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def button2(param):

    print(param)
    df=OpenPoseVideo.OpenPoseVideo(param, OUTPUT, MODE)


def button3(param):
    print('inside button3', param)


def button4(param):
    print('inside button4', param)


def main():
    input_width = 150
    root = Tk()
    root.resizable(width=FALSE, height=FALSE)
    root.geometry("630x700")

    headerinputtext = Label(root, text='Bio mechanic analyzer', anchor=W)
    headerinputtext.place(x=10, y=10, height=25)

    video_cutinputtext = Label(root, text='Video to pictures', anchor=W)
    video_cutinputtext.place(x=10, y=70, width=250, height=20)
    video_cutinput = Entry(root)
    video_cutinput.place(x=10, y=100, width=70 + (input_width + 20) * 2, height=25)
    video_cut_button = Button(root, text='Go', command=lambda: video_cut(video_cutinput.get()))
    video_cut_button.config(relief=RAISED)
    video_cut_button.place(x=450, y=100, width=100, height=25)

    button2inputtext = Label(root, text='Analyze points from video', anchor=W)
    button2inputtext.place(x=10, y=130, width=250, height=25)
    button2input = Entry(root)
    button2input.place(x=10, y=160, width=70 + (input_width + 20) * 2, height=25)
    goButton2 = Button(root, text='Go', command=lambda: button2(button2input.get()))
    goButton2.config(relief=RAISED)
    goButton2.place(x=450, y=160, width=100, height=25)

    button3inputtext = Label(root, text='Predict', anchor=W)
    button3inputtext.place(x=10, y=190, width=250, height=25)
    button3input = Entry(root)
    button3input.place(x=10, y=220, width=70 + (input_width + 20) * 2, height=25)
    goButton3 = Button(root, text='Predict', command=lambda: button3(button3input.get()))
    goButton3.config(relief=RAISED)
    goButton3.place(x=450, y=220, width=100, height=25)

    button4inputtext = Label(root, text='Chart', anchor=W)
    button4inputtext.place(x=10, y=250, width=250, height=25)
    button4input = Entry(root)
    button4input.place(x=10, y=280, width=70 + (input_width + 20) * 2, height=25)
    goButton4 = Button(root, text='Chart', command=lambda: button4(button4input.get()))
    goButton4.config(relief=RAISED)
    goButton4.place(x=450, y=280, width=100, height=25)

    ExitButton = Button(root, text='Exit', command=root.destroy)
    ExitButton.config(relief=RAISED)
    ExitButton.place(x=500, y=400, width=100, height=25)

    root.mainloop()


if __name__ == '__main__':
    main()
