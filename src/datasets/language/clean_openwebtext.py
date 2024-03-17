import os

def remove_empty_lines():
    files = os.listdir('*.txt')

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
        
        with open(file, 'w') as f:
            for line in lines:
                if line.strip():
                    f.write(line)

def join_files():
    files = os.listdir('*.txt')
    with open('openwebtext.txt', 'w') as f:
        for file in files:
            with open(file, 'r') as f2:
                f.write(f2.read())

if __name__ == "__main__":
    print("Cleaning OpenWebText dataset...")
    remove_empty_lines()
    print("Joining files...")
    join_files()