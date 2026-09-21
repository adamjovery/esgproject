import numpy as np
import matplotlib.pyplot as plt
#Function which plots a fan chart for a single simulated variable
def fan_chart(paths, T, ax, title, ylabel, as_percent=False):
    steps = paths.shape[0] - 1
    time_axis = np.linspace(0, T, steps + 1)
 
    #Finding the different percentiles across paths at each time step
    percentile5 = np.percentile(paths, 5, axis=1)
    percentile25 = np.percentile(paths, 25, axis=1)
    percentile50 = np.percentile(paths, 50, axis=1)
    percentile75 = np.percentile(paths, 75, axis=1)
    percentile95 = np.percentile(paths, 95, axis=1)
 
    #Plotting the outer percentiles in a lighter shade
    ax.fill_between(time_axis, percentile5, percentile95, color='steelblue', alpha=0.2, label='5th-95th percentile')
    #Plotting the inner percentiles in a darker shade
    ax.fill_between(time_axis, percentile25, percentile75, color='steelblue', alpha=0.4, label='25th-75th percentile')
    #Plotting the median path
    ax.plot(time_axis, percentile50, color='navy', linewidth=2, label='Median')

    #Creating the axis titles, legend and grid
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Years')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.25)

    #Formatting for percentage if it is a percentage based variables
    if as_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
 
#Function to build a 3-panel fan chart figure for the esg results
def plot_esg_fan_charts(results, T, save_path=None):
    #Intialising a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    #Calling fan chart function for the three variables
    fan_chart(results["rates"], T, axes[0], "Interest Rate (Vasicek)", "Interest rate", as_percent=True)
    fan_chart(results["inflation"], T, axes[1], "Inflation Rate (Ornstein-Uhlenbeck)", "Inflation rate", as_percent=True)
    fan_chart(results["equity"], T, axes[2], "Equity Level (log-GBM)", "Index level", as_percent=False)

    #Arranging the fan charts on a single figure
    fig.suptitle("Economic Scenario Generator: Simulated Paths", fontsize=14, fontweight='bold')
    fig.tight_layout()

    #Checking if the figure is due to be saved and saving it if so
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFan chart saved to {save_path}")
        
    #Returning the completed figure
    return {"overall_chart":fig}
 