# Font Recognition Project

## Overview

This project aims to develop a font recognition system using machine learning techniques. The system identifies and classifies fonts from images of text. It leverages a variety of image processing and classification methods to achieve accurate font recognition.

## Project Structure

The project directory is organized as follows:

. ├── DSA │ 
    ├── README.md │ 
    ├── max_pooling.py 
  ├── Dataset │ 
    └── images_fonts │   
    └── images │ 
    ├── Arimo-Regular │ 
    ├── Dancing+Script-Regular │ 
    ├── FredokaOne-Regular │ 
    ├── NotoSans-Regular │ 
    ├── Open+Sans-Regular │ 
    ├── Oswald-Regular │ 
    ├── PTSerif-Regular │ 
    ├── PatuaOne-Regular │ 
    ├── Roboto-Regular │ 
    └── Ubuntu-Regular 
  ├── Training │ 
    └── niqo-robotics-cv-internship-task.ipynb 
  ├── init.py 
  ├── app.py 
  ├── requirements.txt 
  └── segment.py

## Installation

To set up the project locally:

1. Clone the repository: `git clone https://github.com/smruthi-sumanth/Niqo-Robotics.git`

2. Install required dependencies:
   `pip install -r requirements.txt`

3. Navigate to the project directory:
   `cd Niqo-Robotics`

## Usage

To run the font recognition model:

1. Ensure you have the necessary image files ready
2. Run the jupyter notebook under 'Training' folder to create the .pkl file
3. Run the main script:
   `streamlit run app.py`
