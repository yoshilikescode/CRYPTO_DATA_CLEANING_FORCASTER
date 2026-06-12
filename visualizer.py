import matplotlib.pyplot as plt


def plot_index(index_df):
    # Create a figure with a specific size
    plt.figure(figsize=(18, 6))

    # Plot the index values
    plt.plot(index_df.index, index_df["Index"],
             color="blue", linewidth=1)

    # Add labels and title
    plt.title("Crypto Portfolio Index", fontsize=20)
    plt.xlabel("Date", fontsize=13)
    plt.ylabel("Index Value", fontsize=13)

    # Add a grid for readability
    plt.grid(True, alpha=0.3)

    # Rotate dates so they don't overlap
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()