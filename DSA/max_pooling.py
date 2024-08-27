import numpy as np

def calculate_padding(input_height, input_width, pool_size, stride):
    pool_height, pool_width = pool_size
    stride_height, stride_width = stride

    # Padding calculation
    padding_height = max((input_height - 1) * stride_height + pool_height - input_height, 0)
    padding_width = max((input_width - 1) * stride_width + pool_width - input_width, 0)

    padding_top = padding_height // 2
    padding_bottom = padding_height - padding_top

    padding_left = padding_width // 2
    padding_right = padding_width - padding_left

    return padding_top, padding_bottom, padding_left, padding_right

def pad_matrix(input_array, padding):
    """Pads the input array with specified padding using nested loops."""
    padding_top, padding_bottom, padding_left, padding_right = padding
    input_height, input_width = input_array.shape
    padded_height = input_height + padding_top + padding_bottom
    padded_width = input_width + padding_left + padding_right
    
    # Create a new padded array of zeros
    padded_array = np.zeros((padded_height, padded_width), dtype=input_array.dtype)

    # Copy the original array into the padded array
    for i in range(input_height):
        for j in range(input_width):
            padded_array[i + padding_top, j + padding_left] = input_array[i, j]
    
    return padded_array

def max_pooling_2d(input_array, pool_size=(2, 2), stride=(1, 1)):
    input_height, input_width = input_array.shape
    pool_height, pool_width = pool_size
    stride_height, stride_width = stride

    # Calculate output dimensions
    output_height = (input_height - pool_height) // stride_height + 1
    output_width = (input_width - pool_width) // stride_width + 1

    # Initialize the output array
    pooled_array = np.zeros((output_height, output_width))

    # Perform the max pooling operation with nested loops
    for i in range(output_height):
        for j in range(output_width):
            # Calculate the start and end indices for the pooling window
            start_y = i * stride_height
            end_y = start_y + pool_height
            start_x = j * stride_width
            end_x = start_x + pool_width

            # Initialize the maximum value
            max_value = -float('inf')
            
            # Extract the pooling window and compute the maximum value with nested loops
            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    if input_array[y, x] > max_value:
                        max_value = input_array[y, x]
            
            pooled_array[i, j] = max_value
    
    return pooled_array

# Function to get array from user input
def get_array_from_user():
    # Input dimensions
    rows = int(input("Enter the number of rows in the matrix: "))
    cols = int(input("Enter the number of columns in the matrix: "))

    # Input matrix elements
    print("Enter the elements row by row, separated by spaces:")
    matrix = []
    for _ in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            raise ValueError(f"Row must have exactly {cols} columns.")
        matrix.append(row)
    
    return np.array(matrix)

# Get user inputs
input_array = get_array_from_user()

# Input pooling parameters
pool_size = tuple(map(int, input("Enter the pooling size (height width): ").split()))
stride = tuple(map(int, input("Enter the stride size (height width): ").split()))

# Calculate padding values
input_height, input_width = input_array.shape
padding = calculate_padding(input_height, input_width, pool_size, stride)

# Pad the matrix
padded_array = pad_matrix(input_array, padding)

# Apply max pooling
pooled_array = max_pooling_2d(padded_array, pool_size, stride)

print("Input Array:")
print(input_array)
print("Padded Array:")
print(padded_array)
print("Pooled Array:")
print(pooled_array)