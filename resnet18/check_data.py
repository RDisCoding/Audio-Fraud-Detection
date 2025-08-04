import pickle
import os

# Check batch file structure
batch_file = 'data/preprocessed/resnet/training_batch_0.pkl'
if os.path.exists(batch_file):
    with open(batch_file, 'rb') as f:
        data = pickle.load(f)
    print('Keys in batch file:', list(data.keys()))
    if 'data' in data:
        print('Data shape:', data['data'].shape)
        print('Data type:', type(data['data']))
    if 'labels' in data:
        print('Labels length:', len(data['labels']))
        print('Sample labels:', data['labels'][:5])
        print('Labels type:', type(data['labels']))
else:
    print(f'File {batch_file} not found')

# Count total files
import glob
train_files = glob.glob('data/preprocessed/resnet/training_batch_*.pkl')
val_files = glob.glob('data/preprocessed/resnet/validation_batch_*.pkl')

print(f'\nTotal training batch files: {len(train_files)}')
print(f'Total validation batch files: {len(val_files)}')
