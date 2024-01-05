[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/YPjoen1X)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-7f7980b617ed060a017424585567c406b6ee15c891e84e1186181d67ecf80aa0.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=12775590)
# CS 4337 - Assignment 9 - Face Detection with a Convolutional Neural Network (CNN)

To complete this assignment you will need to implement the following functions:
- `train_face_model()` - Task 1
- `cnn_face_search()` - Task 2

Read the detailed assignment description on Canvas for more information about the functions.

## File list
- `README.md` - This file
- `.gitignore`: This file tells git which files to ignore.
- `train_face_model.py`: This file contains the function `train_face_model()` that you need to implement.
- `cnn_face_search.py`: This file contains the function `cnn_face_search()` that you need to implement.
- `draw_rectangle.py`: This file contains the function `draw_rectangle()` that draws a rectangle on an image.
- `main.py`: This file contains the code that you can use to test your functions.
- `requirements.txt`: This file contains the list of Python packages required to run the code in this repository.
- `data/` - Directory containing the training dataset of faces and non-faces, as well as the test images.
- `tests/` - Directory containing the unit tests for your functions.
- `output/` - Directory where the test output data will be saved.
- `.github/workflows`: This folder contains the GitHub Actions workflow file that is used to run the unit tests on every commit.

## Setting up the environment

If you are running the code on your own machine, and you followed the tutorial posted on Canvas for setting up your environment for running computer vision applications, your environment should be ready to run the code. If you use Github Codespaces the environment will need to be set up by installing the packages listed in the `requirements.txt` file. You can do this by running the following command in the terminal:

```bash
pip install -r requirements.txt
```

**Note:** If on Github Codespaces, after installing the required packages, you get the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`, you will need to run the following command in the terminal:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
```

## Running the code

You can modify the `main.py` file to test your functions. You can run your code using the following command:

```bash
python main.py
```
## Running the tests

To run the unit tests, you will need to run the following command:

```bash
pytest
```

