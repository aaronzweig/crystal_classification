import sys
with open(sys.argv[0]) as f:
    content = f.readlines()
    print("val")
    mean = str(round(float(content[1]), 2))
    var = str(round(float(content[2]), 2))
    print(mean + " $\\pm$ " + var)
    print("train")
    mean = str(round(float(content[4]), 2))
    var = str(round(float(content[5]), 2))
    print(mean + " $\\pm$ " + var)