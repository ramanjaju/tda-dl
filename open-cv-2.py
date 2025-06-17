# import torch
# print(torch.__version__)
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
print(mp.__version__)  
import requests
import zipfile
from pathlib import Path
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_path = Path("data/")
image_path = data_path / "RPS_DATA2"

if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path /"RPS_DATA2.zip" , "wb") as f:
        request = requests.get("https://github.com/hana-4/acmw_ws/raw/main/RPS_DATA2.zip")
        print("Downloading rock, paper, scissors data...")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "RPS_DATA2.zip", "r") as zip_ref:
        print("Unzipping rock, paper, scissors data...")
        zip_ref.extractall(image_path)

import shutil
from pathlib import Path
# Define the path to the _MACOSX folder
macosx_folder = image_path / "__MACOSX"
# Check if the folder exists and delete it
if macosx_folder.is_dir():
    print("Found _MACOSX folder, deleting...")
    shutil.rmtree(macosx_folder)
    print("_MACOSX folder deleted.")
else:
    print("No _MACOSX folder found.")

# Setup train and testing paths
train_dir = image_path /"RPS_DATA2/train"
test_dir = image_path / "RPS_DATA2/test"

train_dir, test_dir

import random
from PIL import Image
import cv2 as cv

# Set seed
random.seed(123)

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*/*.jpg"))
random_image_paths = random.sample(image_path_list, 9)

# Create a plot
fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # 3x3 grid
fig.suptitle("Random Images with Class Labels", fontsize=16)

# Iterate through the selected images and subplots
for ax, random_image_path in zip(axes.flatten(), random_image_paths):
    # 2. Get image class
    image_class = random_image_path.parent.stem

    # 3. Open image
    img = Image.open(random_image_path)

    # Display image in subplot
    ax.imshow(img)
    ax.axis("off")  # Hide axis
    ax.set_title(image_class, fontsize=10)  # Set the title as the class label

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(3, 3))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False)

# Write transform for image
data_transform = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

def plot_transformed_images(image_paths, transform, n=3, seed=123):

    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image, cmap='gray')
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=1)

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

# Get class names as a list
class_names = train_data.classes
class_names

# Can also get class names as a dict
class_dict = train_data.class_to_idx
class_dict

# Check the lengths
len(train_data), len(test_data)

type(train_data[0])

img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")

from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=16, # how many samples per batch?

                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=16,

                             shuffle=False) # don't usually need to shuffle testing data

train_dataloader, test_dataloader

img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

if torch.cuda.is_available():
  device="cuda"
else:
  device="cpu"

class RPSModule(nn.Module):
    def __init__(self):
        super(RPSModule, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=1), # 20
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=5, padding=1), # 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 64, kernel_size=5, padding=1), # 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(64*14*14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.net(x)
    
img = torch.randn(1, 1, 128, 128)  
conv = nn.Conv2d(1, 32, kernel_size=5, padding=1)
maxpool = nn.MaxPool2d(2, 2)
y=conv(img)
y=maxpool(y)
print(y.shape)

from torchmetrics import Accuracy
accuracy_fn=Accuracy(task="multiclass", num_classes=3)

torch.manual_seed(42)
model_1=RPSModule()
optimiser=torch.optim.SGD(params=model_1.parameters(), lr=0.001)
loss_fn=nn.CrossEntropyLoss()

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU (if available)
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_pred.argmax(dim=1),y) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
        test_loss, test_acc = 0, 0
        model.to(device)
        model.eval() # put model in eval mode
        # Turn on inference context manager
        with torch.inference_mode():
            for X, y in data_loader:
                # Send data to GPU
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred = model(X)

                # 2. Calculate loss and accuracy
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(
                    test_pred.argmax(dim=1), y # Go from logits -> pred labels
                )

            # Adjust metrics and print out
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

torch.manual_seed(123)

from tqdm.auto import tqdm

# Train and test model
epochs = 3
for epoch in tqdm(range(epochs)):
 train_step(model_1, train_dataloader, loss_fn, optimiser, accuracy_fn, device)
 test_step(test_dataloader, model_1,loss_fn, accuracy_fn, device)

 # Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model = RPSModule()

import requests
from pathlib import Path
import torch

# Define the URL and local file path
url = "https://github.com/hana-4/acmw_ws/raw/main/rps_model_weights.pth"
save_dir = Path("./weights")
save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
weights_path = save_dir / "rps_model_weights.pth"

# Download the weights file if it doesn't exist
if not weights_path.exists():
    print(f"Downloading weights from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(weights_path, "wb") as f:
            f.write(response.content)
        print(f"Weights downloaded and saved to {weights_path}")
    else:
        print(f"Failed to download weights: {response.status_code}")
else:
    print(f"Weights file already exists at {weights_path}")

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model.load_state_dict(torch.load(f='weights/rps_model_weights.pth'))

import torch
import cv2
import numpy as np

# Load your PyTorch model
model = loaded_model # Replace with your actual loaded model
model.eval()  # Set the model to evaluation mode

# Get the expected input size from the model
input_model_size = (128, 128)  # Update this based on your model's input size

# Define class labels
class_labels = {0: 'paper', 1: 'rock', 2: 'scissors'}

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow('Rock Paper Scissors')

# Define ROIs for Player 1 and Player 2
roi1_x, roi1_y, roi1_w, roi1_h = 200, 150, 300, 300 # ROI for Player 1
roi2_x, roi2_y, roi2_w, roi2_h = 800, 150, 300, 300  # ROI for Player 2

def determine_winner(player1_choice, player2_choice):
    if player1_choice == player2_choice:
        return "Draw"
    elif (player1_choice == 'rock' and player2_choice == 'scissors') or \
         (player1_choice == 'scissors' and player2_choice == 'paper') or \
         (player1_choice == 'paper' and player2_choice == 'rock'):
        return "Player 1 Wins!"
    else:
        return "Player 2 Wins!"

while True:
    try:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Extract ROIs for both players
        roi1 = frame[roi1_y:roi1_y+roi1_h, roi1_x:roi1_x+roi1_w]
        roi2 = frame[roi2_y:roi2_y+roi2_h, roi2_x:roi2_x+roi2_w]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        def predict_choice(roi):
            """Predicts the class label for a given ROI."""
            # Convert ROI to grayscale, resize, normalize, and convert to tensor
            roi_resized = cv2.resize(roi, input_model_size, interpolation=cv2.INTER_AREA)
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            roi_normalized = roi_gray / 255.0
            roi_tensor = torch.tensor(roi_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Classify the ROI image using the PyTorch model
            with torch.no_grad():
                prediction_prob =(model(roi_tensor)).squeeze(0).cpu().numpy()
                #prediction_prob=model(roi_tensor)

            predicted_class_index = np.argmax(prediction_prob)
            predicted_class_label = class_labels[predicted_class_index]
            predicted_class_prob = prediction_prob[predicted_class_index]

            if predicted_class_prob < 0.9:
                return 'Undefined'
            else:
                return predicted_class_label


        # Predict choices for both players
        player1_choice = predict_choice(roi1)
        player2_choice = predict_choice(roi2)

        # Determine winner
        if player1_choice != 'Undefined' and player2_choice != 'Undefined':
            result = determine_winner(player1_choice, player2_choice)
        else:
            result = "Waiting for valid input..."

        # Display ROIs and predictions
        cv2.rectangle(frame, (roi1_x, roi1_y), (roi1_x + roi1_w, roi1_y + roi1_h), (255, 0, 0), 2)
        cv2.putText(frame, f'P1: {player1_choice}', (roi1_x, roi1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.rectangle(frame, (roi2_x, roi2_y), (roi2_x + roi2_w, roi2_y + roi2_h), (0, 255, 0), 2)
        cv2.putText(frame, f'P2: {player2_choice}', (roi2_x, roi2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the result
        cv2.putText(frame, result, (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the frame
        cv2.imshow('Rock Paper Scissors', frame)

    except Exception as e:
        print(e)
        break

cap.release()
cv2.destroyAllWindows()

import cv2 as cv 
import mediapipe as mp

#pip install mediapipe
#pip install opencv-python

#part 1
mp_hands = mp.solutions.hands
#this module contains hand tracking solutions, detects and tracks hand landmarks and connections. uses ml models to identify 21 3d landmarks 
mp_draw = mp.solutions.drawing_utils 
#to draw the landmarks and the connections identified on the images.
mp_draw_styles =  mp.solutions.drawing_styles 
#to get the landmark connection drawing on the image in the defauolt styling (color, thinckness ets)


#part 2 

def hands_movement (hand_landmarks): 
    #made a function takes, landmarks detected as the arguments 
    lm = hand_landmarks.landmark
    #hand_landmarks is an object and youre extracting the landmark attribute of it. list of 21 attributes. 
    #we'll have x,y,z coordinates for everylandmark detected. Thats how we store landmark positions. 
    tips = [lm[i].y for i in [8,12,16,20]]
    #created a list containing of y coordinates of the fingertips
    #the indices correspond to the 4 finger tips

    base = [lm[i-2].y for i in [8,12,16,20]]
    #indices correspond to the same fingertips as before 
    #we're accessing i-2 so it retrieves y coords of the joints below the tips thus the bases


    #making gestures/gesture detection 

    if all(tips[i]>base[i] for i in range(4)):
        return "rock"
    elif all(tips[i]<base[i] for i in range(4)): 
        return "paper"
    elif all(tips[0]<base[0] and tips[1]<base[1] and tips[2]>base[2] and tips[3]>base[3] for i in range (4)):
        return "scissors" 
    
    return "unknown"


#part 3, creating game variables
vid_obj = cv.VideoCapture(0)
#object init to read vdo frames. 
clock = 0
#keep track of the frame count
player1_move = player2_move = None 
gametext = ""
success = True
#weather something has been detected or not



#part 4

#init mediapipe's hand model 
#mp is also ml trained. you can decide how complex you want your model to be. The more complex the model, better the accuracy more the computation
with mp_hands.Hands(model_complexity = 1, 
                    min_detection_confidence = 0.7, 
                    min_tracking_confidence = 0.7) as hands : 

#below two set minimum confidence thresholds for detecting and tracking hands 
    while True : 
        ret, frame = vid_obj.read()

        frame = cv.resize(frame, (1280,720))

        if not ret or frame is None : 
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #mediapipe expects rgb and open cv reads in bgr
        results = hands.process(frame)
        #processing the frame to detect and track the hands 
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        #converting back to bgr format for opencv

        if results.multi_hand_landmarks : 
            #list containing of all detected hands in current frame. so 'if' checks if any landmarks detected, if list empty move on
            for hand_landmark in results.multi_hand_landmarks : 
                #loop iterates over each set of detected hand landmarks
                

                #drawing the landmarks and the connections
                mp_draw.draw_landmarks (
                    frame, 
                    hand_landmark, 
                    mp_hands.HAND_CONNECTIONS, 
                    mp_draw_styles.get_default_hand_landmarks_style(), 
                    mp_draw_styles.get_default_hand_connections_style()
                )
        frame = cv.flip(frame, 1)


        #part5 gamelogic
        #we make the game count on a 100 frame cycle 

        #frame 0 to 19 display players "ready" 
        #frame 20 to 59 --> countdown
        #frame 60 to 99 deteermined the game outcomes 

        if 0<=clock<20 : 
            success = True
            gametext = "ready?"
            #frame 20 to 59 --> countdown
        elif clock<30:
            gametext= "3.."
        elif clock<50: 
            gametext = "2.."
        elif clock<60: 
            gametext = "1, go!..."
        elif clock ==60: 
            #detect hand landmarks and players moves. If two hands are detected success is set to true else false
            hls = results.multi_hand_landmarks
            #retrieves list of detected hand landmarks from results object 
            if hls and len(hls)==2: # to check if two hands detected 
                player1_move = hands_movement(hls[0])  
                player2_move = hands_movement(hls[1])
                #determines players hand movemenets based on first and second set of landmarks 
            else:
                success = False
        elif clock<100:
            #frame 60 to 99 deteermined the game outcomes 
            if success : 
                #if hand detected 
                gametext = f"player 1 played : {player1_move}, Player 2 played: {player2_move}."
                if player2_move == player1_move :
                    gametext = f"{gametext} game is tied"
                elif player1_move == "paper" and player2_move == "rock" : 
                    gametext=f"{gametext} Player 1 Wins"
                elif player1_move == "rock" and player2_move == "scissors" : 
                    gametext = f"{gametext} Player 1 Wins"
                elif player1_move == "scissors" and player2_move =="paper" : 
                    gametext=f"{gametext} Player 1 Wins"
                else:
                    gametext=f"{gametext}Player 2 wins"
            else: 
                gametext = "didnt play properly"

        cv.putText(frame,f"Clock:{clock}",(50,50),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv.LINE_AA)
        cv.putText(frame,gametext,(50,80),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv.LINE_AA)
        clock=(clock+1)%100
        cv.putText(frame, "Press 'Q' to Quit", (50, 120), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow('frame', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            print("Exiting Program...")
            break
vid_obj.release()
cv.destroyAllWindows()
        






