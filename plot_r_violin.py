import random
import statistics

# 20 Zufallswerte zwischen 0 und 1.7, zwei Nachkommastellen
values = [round(random.uniform(0, 2.7), 2) for _ in range(20)]

# Kennzahlen
median_val = statistics.median(values)
min_val = min(values)
max_val = max(values)
std_dev = statistics.stdev(values)

# Latex-konforme Ausgabe
output = (
    f"{median_val:.2f}\\%/"
    f"{min_val:.2f}\\%"
    f"{max_val:.2f}\\%\\\\"
    f"{std_dev:.2f}\\%"
)

print(output)