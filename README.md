# LSTM Electricity Consumption Prediction

## 📌 Project Overview

This is a beginner-level Deep Learning project that demonstrates how to build an **LSTM (Long Short-Term Memory) neural network** using TensorFlow and Keras.

The project uses a simple artificial electricity consumption dataset to understand how LSTM learns patterns from sequential data and predicts the next value.

The main purpose of this project is to understand the fundamentals of LSTM, time series forecasting, sequence creation, data normalization, model training, and prediction.

## 🎯 Project Objectives

- Understand the basics of LSTM neural networks.
- Learn how to prepare sequential data for LSTM.
- Understand time steps, samples, and features.
- Apply MinMaxScaler for data normalization.
- Build an LSTM model using TensorFlow and Keras.
- Train a neural network using historical values.
- Predict the next electricity consumption value.
- Visualize training loss and prediction results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations and array manipulation |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Data normalization |
| TensorFlow | Deep Learning framework |
| Keras | Building and training the LSTM model |

## 📂 Project Structure

```text
LSTM-Electricity-Prediction/
│
├── lstm_electricity_prediction.ipynb
│
├── README.md
│
└── requirements.txt
```

## 📊 Dataset Description

This project uses a manually created artificial electricity consumption dataset.

The dataset contains 20 numerical values representing electricity consumption.

Example:

```python
data = [
    100, 110, 120, 130, 140,
    150, 160, 170, 180, 190,
    200, 210, 220, 230, 240,
    250, 260, 270, 280, 290
]
```

### Dataset Characteristics

- Total observations: 20
- Data type: Numerical
- Pattern: Increasing sequence
- Dataset source: Manually created
- Target: Predict the next electricity consumption value

**Note:** This is an artificial dataset created for learning purposes. It is not real electricity consumption data.

## 🧠 Understanding LSTM

LSTM stands for Long Short-Term Memory.

It is a type of Recurrent Neural Network (RNN) designed to learn patterns from sequential data.

LSTM is commonly used for:

- Time series forecasting
- Electricity consumption prediction
- Stock price analysis
- Weather forecasting
- Sequential data analysis

In this project, the model uses the previous three electricity consumption values to predict the next value.

### Example

```text
Input                  Target

[100, 110, 120]  --->  130

[110, 120, 130]  --->  140

[120, 130, 140]  --->  150
```

## ⚙️ Project Workflow

The project follows these steps:

1. Import required libraries.
2. Create an artificial electricity consumption dataset.
3. Normalize the dataset using MinMaxScaler.
4. Create input sequences and target values.
5. Prepare data in the required LSTM input format.
6. Build the LSTM neural network.
7. Compile the model using Adam optimizer and Mean Squared Error.
8. Train the model.
9. Predict the next electricity consumption value.
10. Convert the prediction back to the original scale.
11. Visualize training loss and prediction results.

## 🔄 Data Preprocessing

### Data Normalization

MinMaxScaler is used to transform the dataset into a range between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(
    data.reshape(-1, 1)
)
```

Normalization helps keep the numerical values within a consistent range during model training.

### Sequence Creation

The model uses three previous values to predict the next value.

```text
Input Shape:

(samples, time_steps, features)

Example:

(17, 3, 1)
```

Where:

- Samples: Number of input sequences
- Time steps: Number of previous values
- Features: Number of input variables

## 🏗️ LSTM Model Architecture

The model is created using TensorFlow and Keras.

```python
model = Sequential()

model.add(
    LSTM(
        50,
        activation="tanh",
        input_shape=(3, 1)
    )
)

model.add(Dense(1))
```

### Model Components

| Layer | Description |
|---|---|
| LSTM | Learns patterns from sequential data |
| Dense | Produces the final numerical prediction |

### Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)
```

- Optimizer: Adam
- Loss function: Mean Squared Error
- Output: One predicted numerical value

## 🚀 Model Training

The model is trained using the prepared sequences.

```python
history = model.fit(
    X,
    y,
    epochs=300,
    batch_size=4,
    verbose=1
)
```

During training, the model learns patterns from the input sequences and updates its weights to reduce prediction error.

## 🔮 Prediction

The trained model uses the last three electricity consumption values to predict the next value.

Example input:

```python
last_values = [270, 280, 290]
```

Expected next value based on the artificial dataset:

```text
300
```

The model prediction is converted back to the original data scale using MinMaxScaler.

The actual prediction may differ from 300 because the LSTM is trained on a very small dataset.

## 📈 Visualization

The project includes two visualizations:

### 1. Training Loss

Shows how the training loss changes during model training.

### 2. Electricity Consumption Prediction

Displays the original electricity consumption values and the predicted next value.

These visualizations help understand the model training process and prediction output.

## 📚 What I Learned

Through this project, I learned:

- Basics of LSTM neural networks.
- Difference between traditional neural networks and recurrent neural networks.
- How sequential data works.
- How to create input sequences and target values.
- How to reshape data for LSTM.
- How to normalize numerical data.
- How to build an LSTM model using TensorFlow.
- How to compile and train a neural network.
- How to make predictions using a trained model.
- How to convert normalized predictions back to the original scale.
- How to visualize training loss and predictions.

## ⚠️ Limitations

- The dataset contains only 20 artificial observations.
- The model is not trained using real electricity consumption data.
- The dataset follows a simple increasing pattern.
- The model may not predict the expected next value accurately.
- Training loss alone does not measure performance on unseen data.
- The project does not include a separate test dataset.

This project is intended for learning LSTM fundamentals and is not suitable for real-world electricity consumption forecasting.

## 🔧 Future Improvements

Possible improvements for future versions:

- Use a real electricity consumption dataset.
- Perform chronological train-test splitting.
- Add more relevant input features.
- Compare LSTM with SimpleRNN and GRU.
- Evaluate the model using MAE, MSE, and RMSE.
- Improve sequence length and model architecture.
- Perform multi-step forecasting.
- Apply the model to a real-world time series forecasting problem.

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LSTM-Electricity-Prediction.git
```

### Step 2: Open the Project

Open the project folder in Jupyter Notebook, Google Colab, or VS Code.

### Step 3: Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Step 4: Run the Notebook

Open:

```text
lstm_electricity_prediction.ipynb
```

Run all cells in order.

## 👨‍💻 Author

Created as a beginner Deep Learning practice project to understand LSTM neural networks and time series forecasting.

## 📌 Conclusion

This project demonstrates a basic LSTM workflow for predicting the next value in a sequential dataset.

It provides practical experience with data preprocessing, sequence creation, neural network architecture, model training, prediction, and visualization.

The project serves as a foundation for developing more advanced time series forecasting models using real-world datasets.