import numpy
import matplotlib.pyplot as plt

def plot_losses_with_sliding_mean(losses, filename):
    """
    Plots the losses with a sliding mean and saves the plot to a file.
    
    Parameters:
    losses (list or numpy array): Array of loss values.
    filename (str): The filename to save the plot.
    """
    # Convert losses to numpy array if it's not already
    losses = numpy.array(losses)
    
    # Calculate the sliding mean with a window of 10% of the data length
    window_size = max(1, int(len(losses) * 0.02))
    sliding_mean = numpy.convolve(losses, numpy.ones(window_size)/window_size, mode='valid')
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the losses
    plt.plot(losses, label='Losses', color='blue')
    
    # Plot the sliding mean
    shift = int((window_size+1)/2)
    plt.plot(range(shift, sliding_mean.shape[0] + shift), sliding_mean, label=f'Sliding Mean (window = {window_size})', linestyle='--', color='orange')
    
    # Annotate the axes
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Losses and Sliding Mean Over Iterations')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(filename)
    plt.close()