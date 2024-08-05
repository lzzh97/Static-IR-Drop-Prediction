from scipy.spatial.distance import cdist
import numpy as np

def min_distance(X, Y):
    
    # compute the distance between two indices of shape (n, d) and (m, d)
    x = np.zeros((len(X[0]),2))
    y = np.zeros((len(Y[0]),2))
    for i, (a,b) in enumerate(zip(X[0], X[1])):
        x[i] = [a,b]
    for i, (a,b) in enumerate(zip(Y[0], Y[1])):
        y[i] = [a,b]
    return cdist(x, y, 'euclidean').min(axis=1)


# Create a function to extract resistance and node coordinates
def extract_data(line):
    components = line.split()
    if len(components) >= 4 and components[0].startswith('R'):
        resistance = float(components[3])  # Resistance value
        node1_coords = tuple(map(int, np.array(components[1].split('_')[-2:], dtype=float) // 2000))  # Coordinates of node 1
        node2_coords = tuple(map(int, np.array(components[2].split('_')[-2:], dtype=float) // 2000))  # Coordinates of node 2
        return resistance, node1_coords, node2_coords
    else:
        return None, None, None


def get_resistance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Find the grid size (maximum x, y coordinates)
    max_x = 0
    max_y = 0
    for line in lines:
        _, node1_coords, node2_coords = extract_data(line)
        if node1_coords and node2_coords:
            max_x = max(max_x, max(node1_coords[0], node2_coords[0]))
            max_y = max(max_y, max(node1_coords[1], node2_coords[1]))

    # Create a grid for resistance distribution
    resistance_grid = np.zeros((max_x + 1, max_y + 1))
    via_grid = np.zeros((max_x + 1, max_y + 1))

    # Populate the resistance grid
    for line in lines:
        resistance, node1_coords, node2_coords = extract_data(line)
        if resistance and node1_coords and node2_coords:
            if resistance_grid[node1_coords] != 0 and resistance_grid[node2_coords] != 0:
                via_grid[node1_coords] += resistance/2
                via_grid[node2_coords] += resistance/2

            resistance_grid[node1_coords] += resistance / 2
            resistance_grid[node2_coords] += resistance / 2

    # Print or visualize the resistance grid
    return resistance_grid, via_grid