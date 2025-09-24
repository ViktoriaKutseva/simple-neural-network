import numpy as np

# Данные для XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[1],
              [0],
              [0],
              [1]])

# --- Функция активации ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # здесь x уже должен быть значением сигмоиды
    return x * (1 - x)

# --- Инициализация весов ---
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)  # для воспроизводимости
    weights1 = np.random.randn(input_size, hidden_size)   # (2,4)
    weights2 = np.random.randn(hidden_size, output_size)  # (4,1)
    return weights1, weights2

# --- Прямое распространение ---
def forward_pass(X, weights1, weights2):
    hidden_layer_input = np.dot(X, weights1)            # (4x2)*(2x4) → (4x4)
    hidden_layer_output = sigmoid(hidden_layer_input)   # применяем сигмоиду
    output_layer_input = np.dot(hidden_layer_output, weights2)  # (4x4)*(4x1) → (4x1)
    predicted_output = sigmoid(output_layer_input)
    return hidden_layer_output, predicted_output

# --- Обратное распространение ошибки ---
def backward_pass(X, y, hidden_layer_output, predicted_output, weights1, weights2, learning_rate):
    # Ошибка выхода
    output_error = y - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    # Ошибка скрытого слоя
    hidden_layer_error = output_delta.dot(weights2.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Обновляем веса
    weights2 += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights1 += X.T.dot(hidden_layer_delta) * learning_rate

    return weights1, weights2

# --- Параметры ---
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 5000

# --- Обучение ---
weights1, weights2 = initialize_weights(input_size, hidden_size, output_size)

for i in range(epochs):
    hidden_layer_output, predicted_output = forward_pass(X, weights1, weights2)
    weights1, weights2 = backward_pass(X, y, hidden_layer_output, predicted_output, weights1, weights2, learning_rate)

    if i % 1000 == 0:
        error = np.mean(np.abs(y - predicted_output))
        print(f"Эпоха {i}, Ошибка: {error:.6f}")

# --- Результат ---
print("\nФинальные предсказания:")
print(np.round(predicted_output))  # округляем до 0 или 1
print("\nФактические выходы сети:")
print(predicted_output)
