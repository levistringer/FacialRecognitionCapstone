from eigenface import eigenface

def varyNeighbours():
    kVal = list(range(0,21))
    accuracy = []
    for k in kVal:
        accuracy.insert(0,eigenface(k))
    print(accuracy)
    pritn(len(accuracy))

varyNeighbours()