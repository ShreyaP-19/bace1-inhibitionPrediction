
try:
    with open('accuracy_results.txt', 'r', encoding='utf-16') as f:
        print(f.read())
except UnicodeError:
    with open('accuracy_results.txt', 'r', encoding='cp1252') as f:
        print(f.read())
except Exception as e:
    print(f"Error: {e}")
