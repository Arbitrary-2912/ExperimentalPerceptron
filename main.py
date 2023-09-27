"""
A mechanism for collecting perceptron data and displaying outputs from pytorch neural networks with an interface designed with customtkinter
"""
import torch
import customtkinter
import os
import time
import math
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image, ImageDraw

## Basic Multi-Class Classifier With Two Hidden Layers and a Standard ReLU Activation Function
class BasicMultiClassClassifier():

    ## Initialize Classifier
    def __init__(self, data_dimensions: tuple, len_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        ## Initialize Model
        self.model = nn.Sequential(
            nn.Linear(in_features=data_dimensions[0] * data_dimensions[1], out_features=2 * len_classes),
            nn.Linear(in_features=2 * len_classes, out_features=int(8. / 5 * len_classes)),
            nn.Relu(),
            nn.Linear(in_features=int(8. / 5 * len_classes), out_features=len_classes)
        )

    ## Simple Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model(x)


## Convolutional Multi-Class Classifier (Larger Image Sizes)
class ConvolutionalClassifier():

    ## Initialize Classifier
    def __init__(self, data_dimensions: tuple, len_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        ## Initialize Model (with Relatively Tunded Hyperparameters)
        self.model = nn.Sequential(
            nn.Sequential(  # Only 1 input channel
                nn.Conv2d(in_channels=1, out_channels=192, kernel_size=5, stride=1),
                nn.BatchNorm2d(num_features=192),
                nn.ReLU(),
                nn.Dropout2d(p=0.05)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=96, kernel_size=3, stride=1),
                nn.BatchNorm2d(num_features=96),
                nn.ReLU(),
                nn.Dropout2d(p=0.05)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, stride=1),
                nn.BatchNorm2d(num_features=48),
                nn.ReLU(),
                nn.Dropout2d(p=0.05)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Linear(in_features=48, out_features=len_classes),
            nn.Softmax()
        )

    ## Simple Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model(x)


## Perceptron
class Perceptron():

    ## Initializing Perceptron Neural Network
    def __init__(self, data_size: int, data_dimensions: tuple, classes: list, *args, **kwargs) -> None:

        ## Seed Torch
        RANDOM_SEED_A = 519724666
        RANDOM_SEED_B = 420911
        torch.manual_seed(RANDOM_SEED_A - RANDOM_SEED_B)

        ## Configuring Percpetron
        self.classes = classes
        self.data_dimensions = data_dimensions

        ## Initializing Model
        self.model = BasicMultiClassClassifier(data_dimensions=self.data_dimensions,
                                               len_classes=len(classes))  # Can be chosen from any of the above models
        self.data_size = data_size

        ## Defining Loss Function and Optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.model.parameters(), lr=0.005)

        ## Defining Statistics
        self.training_loss = []
        self.accuracy = []

    ## Trains Underlying ML Model (change epochs as per data demands)
    def train(self, epochs=75, batch_size=32) -> None:
        global classes
        print('Training Initiated')
        ## Training Loop
        for epoch in range(1, epochs + 1):
            training_epoch_loss = 0
            validation_epoch_loss = 0
            start = time.time()
            ## Training
            for batch in range(math.ceil((self.data_size - int(self.data_size * validation_partition)) / batch_size)):
                ## Gather Training Data
                X_train = 0
                y_train = 0

                for f in os.listdir(os.path.join(os.getcwd(), "data", "train", f"batch{batch + 0:02}")):
                    with open(os.path.join(os.getcwd(), "data", "train", f"batch{batch + 0:02}", f), 'r') as file:
                        data = json.load(file)
                        array_data = (data['data'])
                        c = (classes[data['class']])

                        if isinstance(self.model, BasicMultiClassClassifier):
                            X_train = torch.from_numpy(np.array(array_data).flatten(order='C')).to(torch.float32)
                        else:
                            X_train = torch.from_numpy(np.array(array_data)).to(torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)

                        y_train = torch.zeros(len(classes)); y_train[c] = 1; # y_train.unsqueeze(dim=1)

                        ## Train Model
                        self.model.model.train()
                        ## Forward Pass
                        y_pred = self.model.model(X_train)
                        ## Calculate Loss
                        loss = self.loss_fn(y_pred, y_train)
                        ## Zero Gradients
                        self.optimizer.zero_grad()
                        ## Backpropogation
                        loss.backward()
                        ## Optimizer Step
                        self.optimizer.step()

                        ## Data Metrics
                        training_epoch_loss += loss.item()
            train_end = time.time()

            ## Validation
            with torch.inference_mode():
                ## Evaluate Model
                self.model.model.eval()

                for f in os.listdir(os.path.join(os.getcwd(), "data", "validation")):
                    with open(os.path.join(os.getcwd(), "data", "validation", f), 'r') as file:
                        data = json.load(file)
                        array_data = (data['data'])
                        c = (classes[data['class']])

                        ## Gather Validation Data
                        if isinstance(self.model, BasicMultiClassClassifier):
                            X_val = torch.from_numpy(np.array(array_data).flatten(order='C')).to(torch.float32)
                        else:
                            X_val = torch.from_numpy(np.array(array_data)).to(torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)

                        y_val = torch.zeros(len(classes)); y_val[c] = 1

                        ## Forward Pass
                        y_val_pred = self.model.model(X_val)
                        ## Calculate Loss
                        val_loss = self.loss_fn(y_val_pred, y_val)

                        ## Data Metrics
                        validation_epoch_loss += val_loss.item()

            validation_end = time.time()

            ## Display Epoch Metrics
            print(
                f'Epoch {epoch + 0:03}: | Training Time: {(train_end - start)}s | Training Loss: {training_epoch_loss} | Validation Time: {(validation_end - train_end) + 0:03}s | Validation Loss: {validation_epoch_loss}')

        ## Export Model
        PATH = os.path.join("models", f"trained_model.pt")
        torch.save(self.model.model.state_dict(), PATH)

    ## Classify Image
    def classify(self, prediction_data: np.ndarray) -> str:
        ## Linearize Data if Needed
        if isinstance(self.model, BasicMultiClassClassifier):
            prediction_tensor = torch.from_numpy(prediction_data.flatten(order='C')).to(torch.float32)
        else:
            prediction_tensor = torch.from_numpy(prediction_data).to(torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)

        ## Utilize NN to Determine Class
        output_tensor = self.model.model(prediction_tensor)
        global classifiers
        return classifiers[torch.argmax(output_tensor)] + f' (confidence: {torch.max(output_tensor)})'

## Interface
try:
## Configure customtkinter
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("dark-blue")

    ## Initialize customtkinter root
    root = customtkinter.CTk()
    root.geometry("920x640")
    root.iconbitmap(os.path.join(os.getcwd(), "interface", "perceptron-icon.ico"))
    # root.iconphoto(False, tkinter.PhotoImage(file=("interface\perceptron-icon.png")))
    root.title("Perceptron")

    ## Initialize Configuration Frame
    configuration_frame = customtkinter.CTkFrame(master=root)
    configuration_frame.pack(padx=10, pady=10, fill="both", expand=True)

    ## Initialize Hidden Perceptron Training Frame
    perceptron_data_frame = customtkinter.CTkFrame(master=root)
    perceptron_data_frame.pack_forget()

    ## Initialize Hidden Perceptron Classification Frame
    perceptron_classification_frame = customtkinter.CTkFrame(master=root)
    perceptron_classification_frame.pack_forget()

    ## Raise Initial Frame
    configuration_frame.tkraise()

    ## General Congifuration ##
    training_data_file = None
    training_data = {}
    perceptron = None

    ## Default Perceptron Canvas Size
    perceptron_canvas_height = '400'
    perceptron_canvas_width = '640'

    ## Blank Initilization of Perceptron_Canvas
    perceptron_canvas = customtkinter.CTkCanvas(master=perceptron_data_frame, bg="white", width=perceptron_canvas_width,
                                                height=perceptron_canvas_height)

    ## Null Initialization of Pillow Conterparts
    pil_perceptron_canvas = None
    pil_drawn_canvas = None

    ## Array Store of Classifiers
    classifiers = []
    classifiers_set = {}
    classifiers_freq = {}
    classes = {}
    classifier_label_text = "Enter Classifiers"

    ## Label: Add Classifiers Prompt
    classifier_label = customtkinter.CTkLabel(master=configuration_frame, font=("Roboto", 24), text="Enter Classifiers")
    classifier_label.pack(padx=20, pady=10)

    ## Entry: Add Classifiers
    classifier_entry = customtkinter.CTkEntry(master=configuration_frame, placeholder_text="Add a Label")
    classifier_entry.pack(padx=20, pady=10)


    ## Adds Classifiers
    def addclassifiers():
        global classifiers

        classifiers.append(classifier_entry.get())
        classifiers_set = set(classifiers)
        classifiers = list(classifiers_set)
        list.sort(classifiers)

        time.sleep(0.001)  # Slight delay

        clt = "Enter Classifiers"
        clt += f" - (Current Classifiers: {classifiers})"

        classifier_label.configure(text=clt)

        root.update_idletasks()


    ## Button: Add Classifiers
    classifier_button = customtkinter.CTkButton(master=configuration_frame, text="Add Classifiers",
                                                command=addclassifiers)
    classifier_button.pack(padx=20, pady=10)

    ## Label: Perceptron Size Prompt
    frame_label = customtkinter.CTkLabel(master=configuration_frame, text="Enter Perceptron Frame Size",
                                         font=("Roboto", 20))
    frame_label.pack(padx=20, pady=10)

    ## Entry: Perceptron Width
    frame_width_entry = customtkinter.CTkEntry(master=configuration_frame, placeholder_text="Width")
    frame_width_entry.pack(padx=20, pady=10)

    ## Entry: Perceptron Height
    frame_height_entry = customtkinter.CTkEntry(master=configuration_frame, placeholder_text="Height")
    frame_height_entry.pack(padx=20, pady=10)


    ## Renders Perceptron: Displays Canvas and Initializes Pillow Counterpart
    def render_perceptron():
        global perceptron_canvas_height
        global perceptron_canvas_width
        global perceptron_canvas

        perceptron_data_frame.tkraise()
        configuration_frame.pack_forget()

        try:
            h = str(int(frame_height_entry.get()))
            w = str(int(frame_width_entry.get()))
        except ValueError:
            ## Default Values
            w = '640'
            h = '400'
            pass

        if (int(h) >= 224 and int(w) >= 224):  # minimum height and width
            perceptron_canvas_height = h
            perceptron_canvas_width = w

        ## Creating TkInter Canvas
        perceptron_canvas = customtkinter.CTkCanvas(master=perceptron_data_frame, bg="white",
                                                    width=perceptron_canvas_width, height=perceptron_canvas_height)
        perceptron_canvas.pack(padx=20, pady=10)
        perceptron_canvas.old_coords = None

        ## Creating PIL image
        global pil_perceptron_canvas, pil_drawn_canvas
        pil_perceptron_canvas = Image.new('1', (int(w), int(h)), 1)  ## Mode '1' corresponds to black and white
        pil_drawn_canvas = ImageDraw.Draw(pil_perceptron_canvas)

        ## Creating Classifier Dropdown
        global classifier_dropdown_value
        global canvas_classifier_dropdown
        global classifiers
        classifier_dropdown_value = customtkinter.StringVar(master=perceptron_data_frame)
        classifier_dropdown_value.set("-")
        canvas_classifier_dropdown = customtkinter.CTkOptionMenu(master=perceptron_data_frame,
                                                                 variable=classifier_dropdown_value, values=classifiers)
        canvas_classifier_dropdown.pack(padx=20, pady=10)

        ## Rendering the Perceptron Training Frame
        perceptron_data_frame.pack(padx=10, pady=10, fill="both", expand=True)


    ## Button: Render Perceptron
    render_perceptron_button = customtkinter.CTkButton(master=configuration_frame, text="Render Perceptron",
                                                       command=render_perceptron)
    render_perceptron_button.pack(padx=20, pady=10)

    ## Label: Perceptron
    label = customtkinter.CTkLabel(master=perceptron_data_frame, text="Perceptron", font=("Roboto", 40))
    label.pack(padx=20, pady=10)

    ## Perceptron Data Collection ##
    x = 0
    y = 0
    x1 = 0
    y1 = 0
    mouse_update = False


    ## Mouse Release Bound Action to Enhance Drawing on Perceptron Canvas
    def on_mouse_release(e):
        global x, y, mouse_update
        if not mouse_update:
            mouse_update = True
            x = e.x
            y = e.y


    ## Mouse Bound Action to Allow Drawing on Perceptron Canvas
    def draw_on_canvas(e):
        global x, y, x1, y1, mouse_update
        if mouse_update:
            x = e.x
            y = e.y
            mouse_update = False
        else:
            x = x1
            y = y1
        if perceptron_canvas.old_coords:
            x1 = e.x
            y1 = e.y
            if (x != 0 and y != 0):
                global pil_drawn_canvas
                perceptron_canvas.create_line(x, y, x1, y1)
                pil_drawn_canvas.line([x, y, x1, y1], 0)
        perceptron_canvas.old_coords = x, y


    ## Clears Perceptron Canvas
    def clear_canvas():
        perceptron_canvas.delete('all')
        global x
        global y
        x = 0
        y = 0
        root.update_idletasks()

        global pil_perceptron_canvas, pil_drawn_canvas
        pil_perceptron_canvas = Image.new('1', (int(perceptron_canvas_width), int(perceptron_canvas_height)), 1)
        pil_drawn_canvas = ImageDraw.Draw(pil_perceptron_canvas)


    ## Button: Clear Canvas
    clear_canvas_frame_button = customtkinter.CTkButton(master=perceptron_data_frame, text="Clear Canvas",
                                                        command=clear_canvas)
    clear_canvas_frame_button.pack(padx=20)

    ## Parses Training Data
    frame_index = 0


    def parse_frame():
        global frame_index  # For file naming purposes

        try:
            data_class = classifier_dropdown_value.get()  # Accessing Selected Class
            if data_class == '-':
                return
        except:
            return

        ## Saving Canvas Image
        file_name = os.path.join("data", f"training_canvas_frame_{data_class}_{frame_index}.jpg")
        pil_perceptron_canvas.save(file_name)

        print(f'Frame Saved As {file_name} And Converted To JSON')

        ## Parsing Canvas Image Into Array
        numpy_data = np.asarray(pil_perceptron_canvas).astype(
            int).tolist()  # white is stored as 1, black is stored as 0

        ## Populating Training Data Into JSON
        global training_data_file
        training_data_file_name = os.path.join(os.getcwd(), 'data', f'training_data_{frame_index}.json')
        training_data_file = open(training_data_file_name, 'x')
        training_data.update({'class': data_class})
        training_data.update({'data': numpy_data})
        json.dump(training_data, training_data_file, indent=4)
        training_data_file.close()

        ## Iterating Frame
        frame_index += 1
        if list(classifiers_freq.keys()).count(data_class) == 0 or classifiers_freq[data_class] is None:
            t = classifiers_freq[data_class] = []
            t.append(training_data_file_name)
            classifiers_freq[data_class] = t
        else:
            t = classifiers_freq[data_class]
            t.append(training_data_file_name)
            classifiers_freq[data_class] = t

        ## Deleting JPEG
        os.remove(file_name)

        ## Clearing Canvas
        clear_canvas()


    ## Button: Add Data
    add_canvas_frame_button = customtkinter.CTkButton(master=perceptron_data_frame, text="Add Data",
                                                      command=parse_frame)
    add_canvas_frame_button.pack(padx=20, pady=10)


    ## Splits Data into Train and Validation Sets
    def split_data(batch_size=32):
        global classes, classifiers, classifiers_freq, validation_partition
        validation_partition = 0.1
        sum = 0
        t_lst = []

        ## Create Validation Split
        for i in classifiers:
            if (list(classifiers_freq.keys()).count(i) > 0):
                f_lst = classifiers_freq[i]
                n = int(validation_partition * len(f_lst))
                v_lst = random.sample(population=f_lst, k=n)
                f_lst = list(filter(lambda j: j not in v_lst, f_lst))
                t_lst.extend(f_lst)
                if (len(v_lst) > 0 and os.listdir(os.path.join(os.getcwd(), 'data')).count(
                        'validation') == 0):
                    os.mkdir(os.path.join('data', 'validation'))
                for f in v_lst:
                    v_spl = f.split(os.sep)
                    v_spl.insert(len(v_spl) - 1, 'validation')
                    sv_spl = os.sep.join(v_spl)
                    os.rename(f, sv_spl)
                classifiers_freq[i] = f_lst
                sum += len(f_lst)

        batch_size = min(sum, batch_size)
        num_batches = math.ceil(float(sum) / batch_size)
        ## Split Training Data Into Batches
        for i in range(num_batches):
            ti_lst = random.sample(population=t_lst, k=min(len(t_lst), batch_size))
            t_lst = list(filter(lambda j: j not in ti_lst, t_lst))
            if len(ti_lst) > 0 and os.listdir(os.path.join(os.getcwd(),'data')).count('train') > 0 and os.listdir(os.path.join(os.getcwd(), 'data', 'train')).count(os.path.join(f'batch{i + 0:02}')) == 0:
                os.chdir(os.path.join(os.getcwd(), 'data')); os.mkdir(os.path.join('train', f'batch{i + 0:02}'))
                os.chdir(''); os.chdir('')
            for f in ti_lst:
                ti_spl = f.split(os.sep)
                ti_spl.insert(len(ti_spl) - 1, 'train')
                ti_spl.insert(len(ti_spl) - 1, f'batch{i + 0:02}')
                ts_spl = os.sep.join(ti_spl)
                os.rename(f, ts_spl)


    ## Trains Perceptron and Brings to Detction Frame
    def train_perceptron():
        ## Generate Classes
        global classes, classifiers, frame_index
        for i in range(len(classifiers)):
            classes[classifiers[i]] = i

        ## Split Data
        split_data()

        ## Train Perceptron
        global perceptron
        perceptron = Perceptron(data_size=frame_index,
                                data_dimensions=(int(perceptron_canvas_width), int(perceptron_canvas_height)),
                                classes=classifiers)
        perceptron.train()

        ## Update Visual Frames
        global perceptron_data_frame, perceptron_classification_frame
        perceptron_data_frame.pack_forget()

        ## Creating TkInter Canvas
        global perceptron_canvas
        perceptron_canvas = customtkinter.CTkCanvas(master=perceptron_classification_frame, bg="white",
                                                    width=perceptron_canvas_width, height=perceptron_canvas_height)
        perceptron_canvas.pack(padx=20, pady=10)
        perceptron_canvas.old_coords = None

        ## Creating PIL image
        global pil_perceptron_canvas, pil_drawn_canvas
        pil_perceptron_canvas = Image.new('1', (int(perceptron_canvas_width), int(perceptron_canvas_height)),
                                          1)  ## Mode '1' corresponds to black and white
        pil_drawn_canvas = ImageDraw.Draw(pil_perceptron_canvas)

        ## Updating Clear Canvas Frame Button
        global clear_canvas_frame_button
        clear_canvas_frame_button = customtkinter.CTkButton(master=perceptron_classification_frame, text="Clear Canvas",
                                                            command=clear_canvas)
        clear_canvas_frame_button.pack(padx=20)

        perceptron_classification_frame.pack(padx=20, pady=20)
        root.update_idletasks()


    ## Button: Train Perceptron
    train_perceptron_button = customtkinter.CTkButton(master=perceptron_data_frame, text="Train Perceptron",
                                                      command=train_perceptron)
    train_perceptron_button.pack(padx=20, pady=10)

    ## Label: Display Class
    class_label_var = customtkinter.StringVar()
    class_label_var.set('Draw and Detect')
    display_class_label = customtkinter.CTkLabel(master=perceptron_classification_frame, textvariable=class_label_var,
                                                 font=("Roboto", 24))
    display_class_label.pack(padx=20, pady=10)


    ## Classifies The Current Perceptron Frame
    def classify_frame():
        ## Saving Current Canvas Image Frame
        file_name = os.path.join("data", f"detection_frame.jpg")
        pil_perceptron_canvas.save(file_name)

        ## Parsing Canvas Image Into Array
        numpy_data = np.asarray(pil_perceptron_canvas).astype(int)  # white is stored as 1, black is stored as 0
        os.remove(file_name)

        ## Passing Image Data Into Perceptron
        global perceptron
        prediction = perceptron.classify(numpy_data)

        ## Displaying Perceptron Prediction
        class_label_var.set(f'Prediction: {prediction}')
        time.sleep(0.001)

        root.update_idletasks


    ## Button: Classify Frame
    classify_button = customtkinter.CTkButton(master=perceptron_classification_frame, text="Classify",
                                              command=classify_frame)
    classify_button.pack(padx=20, pady=10)

    ## Mouse Actions
    root.bind('<Button-1>', on_mouse_release)
    root.bind('<B1-Motion>', draw_on_canvas)  # binds left click to canvas draw action

    ## Root Looper
    root.mainloop()

except (KeyboardInterrupt, BaseException):
    ## Exit Behaviour
    print('Program Forcibly Terminated:')
    print('Deleting Data Files')

    ## Move to the data directory
    os.chdir('data')

    ## Remove all images
    data_files = os.listdir(os.getcwd())
    for file in data_files:
        if file.endswith('.json'):
            os.remove(os.path.join(os.getcwd(), file))

    ## Exit the data directory
    os.chdir('..')

    print('Data Files Deleted')
