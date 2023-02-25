import pandas as pd

def asTable(array, columns):
    data = {}
    index = 0
    for c in columns:
        data[c] = []
        for record in array:
            data[c].append(record[index])
        index = index + 1
    df = pd.DataFrame(data)
    print(df)