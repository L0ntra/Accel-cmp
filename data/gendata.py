import sys
import json
import random

def main(n, m, filename):
  random.seed()
  #Generate random numbers where 0 <= num < 10000
  data = [[random.random() * 10 ** random.randint(0, 4) for j in range(0, m)] for i in range(0, n)]

  #Write to JSON File
  with open(filename, "w") as file:
    json.dump({'data':data}, file, indent=2)  
  file.close()

  print(n, "x", m, " file generaged as '", filename, "'", sep = '')

if __name__ == "__main__":
  if len(sys.argv) == 2:
    main(int(sys.argv[1]), int(sys.argv[1]), sys.argv[1] + 'x' + sys.argv[1] + '.json')
  elif len(sys.argv) == 3:
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[1] + 'x' + sys.argv[2] + '.json')
  else:
    print("Error: Called with 1 or 2 integer arguments")
