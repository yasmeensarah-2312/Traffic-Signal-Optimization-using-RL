import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------

CSV_FILE = os.path.join(
    "results",
    "fixed_time_baseline.csv"
)

OUTPUT_DIR = "results"


# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

data = pd.read_csv(CSV_FILE)

print("\nFixed-Time Baseline Data")
print(data)


# --------------------------------------------------
# METRICS TO PLOT
# --------------------------------------------------

metrics = {
    "total_waiting_time": "Total Waiting Time",
    "average_queue_length": "Average Queue Length",
    "throughput": "Throughput",
    "average_delay": "Average Delay"
}


# --------------------------------------------------
# CREATE GRAPHS
# --------------------------------------------------

for column, title in metrics.items():

    plt.figure()

    plt.bar(
        data["scenario"],
        data[column]
    )

    plt.title(
        f"Fixed-Time Baseline - {title}"
    )

    plt.xlabel("Scenario")
    plt.ylabel(title)

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"fixed_time_{column}.png"
    )

    plt.savefig(output_file)

    plt.close()

    print(f"Saved: {output_file}")


print("\nBaseline graphs generated successfully.")