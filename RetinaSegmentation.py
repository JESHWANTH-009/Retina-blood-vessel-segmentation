from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import cv2
import pickle
from keras.models import model_from_json

main = tkinter.Tk()
main.title("Retina blood vessel segmentation with a convolution neural network (U-net)")
main.geometry("1200x1200")


global model
global filename

def getAlexModel(input_size=(64,64,1)):
    inputData = Input(input_size)
    
    conv01 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputData)
    conv01 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv01)
    pool01 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv02 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool01)
    conv02 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv02)
    pool02 = MaxPooling2D(pool_size=(2, 2))(conv02)

    conv03 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool02)
    conv03 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv03)
    pool03 = MaxPooling2D(pool_size=(2, 2))(conv03)

    conv04 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool03)
    conv04 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv04)
    pool04 = MaxPooling2D(pool_size=(2, 2))(conv04)

    conv05 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool04)
    conv05 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv05)

    up06 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv05), conv04], axis=3)
    conv06 = Conv2D(256, (3, 3), activation='relu', padding='same')(up06)
    conv06 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv06)

    up07 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv06), conv03], axis=3)
    conv07 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv07 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up08 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv07), conv02], axis=3)
    conv08 = Conv2D(64, (3, 3), activation='relu', padding='same')(up08)
    conv08 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv08)

    up09 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv08), conv01], axis=3)
    conv09 = Conv2D(32, (3, 3), activation='relu', padding='same')(up09)
    conv09 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv09)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv09)

    return Model(inputs=[inputData], outputs=[conv10])



def loadModel():
    global model
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights("model/model_weights.h5")
    model._make_predict_function()   
    print(model.summary())
    pathlabel.config(text='U-net model loaded')
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    train_accuracy = data['binary_accuracy']
    validate_accuracy = data['val_binary_accuracy']
    text.delete('1.0', END)
    text.insert(END,"U-net Training Accuracy : "+str(train_accuracy[99])+"\n")
    text.insert(END,"U-net Validation Accuracy : "+str(validate_accuracy[99])+"\n")

def uploadImage():
    global filename
    filename = askopenfilename(initialdir = "testImages")
    pathlabel.config(text=filename)

def segmentation():
    global filename
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = model.predict(img)
    preds = preds[0]
    cv2.imwrite('segment.png',cv2.resize((preds*255),(255,255),interpolation = cv2.INTER_CUBIC))
    figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10,10))
    axis[0].set_title("Original Image")
    axis[1].set_title("Retina Blood Vessel Segmented Image")
    axis[0].imshow(cv2.imread(filename),cmap='gray')
    axis[1].imshow(cv2.imread('segment.png'),cmap='gray')
    figure.tight_layout()
    plt.show()
    

    

    

def graph():
    text.delete('1.0', END)
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    train_accuracy = data['binary_accuracy']
    validate_accuracy = data['val_binary_accuracy']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(train_accuracy, 'ro-', color = 'indigo')
    plt.plot(validate_accuracy, 'ro-', color = 'green')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('U-net Training & Validation Accuracy Comparison Graph')
    plt.show()
        
        

font = ('times', 14, 'bold')
title = Label(main, text='Retina blood vessel segmentation with a convolution neural network (U-net)')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Generate & Load U-net Model", command=loadModel)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=430,y=100)

demoButton = Button(main, text="Upload Retina Image", command=uploadImage)
demoButton.place(x=50,y=150)
demoButton.config(font=font1)


dcpButton = Button(main, text="Retina Segmentation", command=segmentation)
dcpButton.place(x=50,y=200)
dcpButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=400)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
