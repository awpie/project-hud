# project-hud
## What is this?
This is the training/inference/server code for a AR + Computer Vision platform with modular support of various applications requiring different software and models.
## What does this do?
The first iteration of the hardware of this project involves a simple camera module + Wi-Fi server setup that streams camera feed over Wi-Fi to a more powerful device that can run inference on the camera stream.
Once inference is done, the results are sent over Wi-Fi back to the microcontroller hosting the server to allow it to correctly update an OLED module that beams a "Heads-Up-Display" directly onto the user's Field-Of-View.

## Uses
Project HUD is a modular, variable-use platform that can provide a realtime Heads-Up-Display for many different settings, provided that the model being ran is not prohibitively expensive or slow for the device that runs it, and latency requirements of the use case are met with Wi-Fi speeds (or by giving me enough money to fit a powerful computer on the hardware itself).
The "activity detection" module uses OpenCLIP classification to determine if the HUD user is in the process of performing some sort of task, such as eating, studying, exercising, speaking with someone, etc, and the HUD displays a progress bar and time-based completion tracking. Yes, like a video game.
A "hazard detection" module could, for example, be trained on an image dataset containing workplace OSHA violations to quickly identify potential hazards in the user's Field-Of-View and display these hazards on the Heads-Up-Display. A backwards-mounted camera could notify the user of hazards *that they cannot possibly see*. 
An "education module" could, for example, be trained on an dataset with YOLOv8's posture and hand position features extracted. The original dataset could include first-person footage of someone performing a highly specific task correctly (stripping wires when installing an electrical outlet, performing a particular type of hand movement while playing the piano, correct hand positioning to pass a driver's license test), and the model could detect whether the HUD user is performing the particular hand movements that correlate to correctly completing the task, and provide *instant feedback* that hihglights any mistakes.
A "pilot module" could, for example, be trained on dashcam car crash footage to quickly alert a driver if signs of an imminent crash is detected. For example, if the car in front of you slams on their brakes, it would be pretty hard to not notice, *even if you were looking the wrong way*, if your Field-of-View suddenly flashed red. Adding with a microphone, and training a parallel model on sounds of car horns, could allow audio-impaired drivers to "see" if another car is sounding their horn.

## Quickstart
write this later
