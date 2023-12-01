import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Constants
NUM_BALLS = 100
RADIUS = 0.2
MOVES = 25  # Number of moves

# Box dimensions
BOX_WIDTH = 10
BOX_HEIGHT = 10

grid_width = BOX_WIDTH / 10
grid_height = BOX_HEIGHT / 10
ball_positions = [(j * grid_width + grid_width / 2, i * grid_height + grid_height / 2)
                  for i in range(10) for j in range(10)]

# Function to Check Overlap 
def pbc_distance(pos1, pos2):
    dx = np.abs(pos1[0] - pos2[0])
    dy = np.abs(pos1[1] - pos2[1])
    dx = min(dx, BOX_HEIGHT - dx)
    dy = min(dy, BOX_HEIGHT - dy)
    return np.sqrt(dx**2 + dy**2)

# Function to move a single ball
def move_ball(position):
    angle = np.random.rand() * 2 * np.pi
    distance = np.random.exponential(scale=2 * RADIUS)
    new_pos = position + np.array([distance * np.cos(angle), distance * np.sin(angle)])
    new_pos[0] = new_pos[0] % BOX_WIDTH
    new_pos[1] = new_pos[1] % BOX_HEIGHT
    return new_pos

#Function to move all balls once
def move_balls_one_round():
    for i in range(NUM_BALLS):
        new_pos = move_ball(ball_positions[i])
        if not any(pbc_distance(new_pos, p) < 2 * RADIUS for j, p in enumerate(ball_positions) if j != i):
            ball_positions[i] = new_pos


# Include all positions including mirror images
def mirrored_positions():
    mirror_ball_positions = []
    for i in range(NUM_BALLS):
        # Original positions
        mirror_ball_positions.append(ball_positions[i])
        # Mirrored positions
        for x_shift in range(-1, 2): 
            for y_shift in range(-1, 2):
                if x_shift != 0 or y_shift != 0:
                    shifted_pos = np.array([ball_positions[i][0] + x_shift * BOX_WIDTH, 
                                            ball_positions[i][1] + y_shift * BOX_HEIGHT])
                    mirror_ball_positions.append(shifted_pos)
    return mirror_ball_positions

def calculate_distances(pos1, pos2):
    dx = (np.abs((pos1[0])-(pos2[0]))) ** 2
    dy = (np.abs((pos1[1])-(pos2[1]))) ** 2
    return (np.sqrt(dx + dy))

#Initialize the histogram sum
histogram_sums = np.zeros(int(50 * RADIUS / RADIUS))  # Corrected to have a size of 50



# Set up the figure and axes for the plots
fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(15, 6))
plt.ion()  # Turn on interactive mode for live updates

# Initialize the total histogram sums
total_histogram_sums = np.zeros(50)

# Plotting and histogram calculation
for move in range(MOVES):
    move_balls_one_round()
    mirrored_ball_positions = mirrored_positions()

    # Clear previous plots
    ax_box.clear()
    ax_hist.clear()

    # Plot ball positions and label them
    for index, pos in enumerate(ball_positions):
        # Plot and label balls in the central box
        circle = Circle((pos[0], pos[1]), RADIUS, color='blue', fill=False)
        ax_box.add_patch(circle)
        ax_box.text(pos[0], pos[1], str(index + 1), fontsize=6, ha='center')

        # Plot and label mirrored positions
        for x_shift in range(-1, 2): 
            for y_shift in range(-1, 2):
                if x_shift != 0 or y_shift != 0:
                    mirrored_pos = np.array([pos[0] + x_shift * BOX_WIDTH, pos[1] + y_shift * BOX_HEIGHT])
                    circle = Circle((mirrored_pos[0], mirrored_pos[1]), RADIUS, color='blue', fill=False)
                    ax_box.add_patch(circle)
                    ax_box.text(mirrored_pos[0], mirrored_pos[1], str(index + 1), fontsize=6, ha='center')

    # Draw lines to separate the 9 boxes
    for x in range(-BOX_WIDTH, 2 * BOX_WIDTH, BOX_WIDTH):
        ax_box.axvline(x, color='black', linestyle='--')
    for y in range(-BOX_HEIGHT, 2 * BOX_HEIGHT, BOX_HEIGHT):
        ax_box.axhline(y, color='black', linestyle='--')

    # Set plot limits and title
    ax_box.set_xlim(-BOX_WIDTH, 2 * BOX_WIDTH)
    ax_box.set_ylim(-BOX_HEIGHT, 2 * BOX_HEIGHT)
    ax_box.set_title(f'Balls after move {move + 1}')
    # Reset histogram sums for cumulative histogram
    histogram_sums = np.zeros(50)  # Corrected size

    # Calculate distances and update histogram
    for pos in mirrored_ball_positions[1:]:  # Exclude the first ball
        distance = calculate_distances(pos, mirrored_ball_positions[0])
        if distance <= 50 * RADIUS:
            bin_index = int(distance // RADIUS)
            if bin_index < len(histogram_sums):  # Check to avoid index out of bounds
                histogram_sums[bin_index] += 1
                total_histogram_sums[bin_index] += 1  # Add to total sums

    # Plot cumulative histogram
    ax_hist.bar(np.arange(len(histogram_sums)) * RADIUS, np.cumsum(histogram_sums), width=RADIUS)
    ax_hist.set_xlim(0, 50 * RADIUS)
    ax_hist.set_xticks(np.arange(0, 50 * RADIUS, RADIUS))  # Set x-ticks
    ax_hist.set_xticklabels([f'{i}r' for i in range(1, 51)], rotation='vertical')  # Set x-tick labels
    ax_hist.set_title('Cumulative Histogram of Distances')
    ax_hist.set_xlabel('Distance')
    ax_hist.set_ylabel('Cumulative Number of Balls')

    plt.draw()
    plt.pause(0.1)  # Small pause to update plots

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plots

# Calculate the average cumulative histogram
average_histogram_sums = total_histogram_sums / MOVES

# Plot the average cumulative histogram
fig_avg, ax_avg_hist = plt.subplots(figsize=(8, 6))
ax_avg_hist.bar(np.arange(len(average_histogram_sums)) * RADIUS, np.cumsum(average_histogram_sums), width=RADIUS)
ax_avg_hist.set_xlim(0, 50 * RADIUS)
ax_avg_hist.set_xticks(np.arange(0, 50 * RADIUS, RADIUS))
ax_avg_hist.set_xticklabels([f'{i}r' for i in range(1, 51)], rotation='vertical')
ax_avg_hist.set_title('Average Cumulative Histogram of Distances')
ax_avg_hist.set_xlabel('Distance')
ax_avg_hist.set_ylabel('Average Cumulative Number of Balls')

plt.show()  # Show the average histogram

# Calculate average histogram
average_histogram_sums = total_histogram_sums / MOVES

num_bins = 50
# Calculate RDF
rdf = np.zeros(num_bins)
area_of_bins = np.pi * (np.arange(1, num_bins + 1)**2 * RADIUS**2 - np.arange(0, num_bins)**2 * RADIUS**2)
density = NUM_BALLS / (BOX_WIDTH * BOX_HEIGHT)
for i in range(num_bins):
    rdf[i] = (average_histogram_sums[i] / area_of_bins[i]) / density

# Plot RDF
plt.figure()
plt.plot(np.arange(num_bins) * RADIUS, rdf)
plt.xlabel('Distance (r)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
plt.show()