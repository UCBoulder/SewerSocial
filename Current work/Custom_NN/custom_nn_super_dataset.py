import os

# Show only current project folder structure
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    level = root.count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        if f.endswith('.csv') or f.endswith('.py') or f.endswith('.ipynb'):
            print(f'{subindent}{f}')