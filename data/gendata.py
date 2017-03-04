import sys
import json
import random

def main(n, m, filename):
  random.seed()

  file = open(filename, 'w')
  file.write(str(n)); file.write("\n")
  file.write(str(m)); file.write("\n")
  for i in range(0, m):
    for j in range(0, n):
      file.write(str(random.random() * 10 ** random.randint(0, 4)))
      file.write("\n")

  file.close()

  print(m, "x", n, " file generaged as '", filename, "'", sep = '')

if __name__ == "__main__":
  if len(sys.argv) == 2:
    main(int(sys.argv[1]), int(sys.argv[1]), sys.argv[1] + 'x' + sys.argv[1] + '.json')
  elif len(sys.argv) == 3:
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[1] + 'x' + sys.argv[2] + '.json')
  else:
    print("Error: Called with 1 or 2 integer arguments")
