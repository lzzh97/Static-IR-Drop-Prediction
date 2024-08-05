import torch
from torch.utils import data
import os
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms
import re
from collections import defaultdict



# Create a function to extract resistance and node coordinates
def extract_data(line):
    components = line.split()
    if len(components) >= 4 and components[0].startswith('R'):
        resistance = float(components[3])  # Resistance value
        node1_coords = tuple(map(int, np.array(components[1].split('_')[-2:], dtype=float) // 2000))  # Coordinates of node 1
        node2_coords = tuple(map(int, np.array(components[2].split('_')[-2:], dtype=float) // 2000))  # Coordinates of node 2
        return resistance, node1_coords, node2_coords
    else:
        return None, None, None


def get_resistance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Find the grid size (maximum x, y coordinates)
    max_x = 0
    max_y = 0
    for line in lines:
        _, node1_coords, node2_coords = extract_data(line)
        if node1_coords and node2_coords:
            max_x = max(max_x, max(node1_coords[0], node2_coords[0]))
            max_y = max(max_y, max(node1_coords[1], node2_coords[1]))

    # Create a grid for resistance distribution
    resistance_grid = np.zeros((max_x + 1, max_y + 1))
    via_grid = np.zeros((max_x + 1, max_y + 1))

    # Populate the resistance grid
    for line in lines:
        resistance, node1_coords, node2_coords = extract_data(line)
        if resistance and node1_coords and node2_coords:
            if resistance_grid[node1_coords] != 0 and resistance_grid[node2_coords] != 0:
                via_grid[node1_coords] += resistance/2
                via_grid[node2_coords] += resistance/2

            resistance_grid[node1_coords] += resistance / 2
            resistance_grid[node2_coords] += resistance / 2

    # Print or visualize the resistance grid
    return resistance_grid, via_grid


class load_fake(data.Dataset):
    def __init__(self, root):
        maps = os.listdir(root)
        grouped_items = defaultdict(list)
        for m in maps:
            prefix = re.split('_|\.',m)[1]
            grouped_items[prefix].append(m)
            
        grouped_list = list(grouped_items.values())
        for i in range(len(grouped_list)):
            grouped_list[i] = [os.path.join(root, m) for m in grouped_list[i]]
        self.maps = grouped_list
        self.maps.sort(reverse=False)
    
    def __getitem__(self, index):
        collect_m = self.maps[index]
        for m_path in collect_m:
            if '_current.csv' in m_path:
                current = np.genfromtxt(m_path, delimiter = ',')
                current = current/current.max()
                current = torch.tensor(resize(current, [512,512])).float().unsqueeze(0)
            elif 'dist' in m_path:
                dist = np.genfromtxt(m_path, delimiter = ',')
                dist = dist/dist.max()
                dist = torch.tensor(resize(dist, [512,512])).float().unsqueeze(0)
            elif 'pdn' in m_path:
                pdn = np.genfromtxt(m_path, delimiter = ',')
                pdn = pdn/pdn.max()
                pdn = torch.tensor(resize(pdn, [512,512])).float().unsqueeze(0)
            elif 'ir_drop' in m_path:
                ir_drop = np.genfromtxt(m_path, delimiter = ',')
                ir_drop = torch.tensor(resize(ir_drop, [512,512])).float().unsqueeze(0)
                
            # resistances and vias are normalized when generated
            elif 'resistance_m1' in m_path:
                resistance_m1 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m1 = resistance_m1/resistance_m1.max()
                resistance_m1 = torch.tensor(resize(resistance_m1, [512,512])).float().unsqueeze(0)
            elif 'resistance_m4' in m_path:
                resistance_m4 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m4 = resistance_m4/resistance_m4.max()
                resistance_m4 = torch.tensor(resize(resistance_m4, [512,512])).float().unsqueeze(0)
            elif 'resistance_m7' in m_path:
                resistance_m7 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m7 = resistance_m7/resistance_m7.max()
                resistance_m7 = torch.tensor(resize(resistance_m7, [512,512])).float().unsqueeze(0)
            elif 'resistance_m8' in m_path:
                resistance_m8 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m8 = resistance_m8/resistance_m8.max()
                resistance_m8 = torch.tensor(resize(resistance_m8, [512,512])).float().unsqueeze(0)
            elif 'resistance_m9' in m_path:
                resistance_m9 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m9 = resistance_m9/resistance_m9.max()
                resistance_m9 = torch.tensor(resize(resistance_m9, [512,512])).float().unsqueeze(0)
            elif 'via_m1m4' in m_path:
                via_m1m4 = np.genfromtxt(m_path, delimiter = ',')
                via_m1m4 = via_m1m4/via_m1m4.max()
                via_m1m4 = torch.tensor(resize(via_m1m4, [512,512])).float().unsqueeze(0)
            elif 'via_m4m7' in m_path:
                via_m4m7 = np.genfromtxt(m_path, delimiter = ',')
                via_m4m7 = via_m4m7/via_m4m7.max()
                via_m4m7 = torch.tensor(resize(via_m4m7, [512,512])).float().unsqueeze(0)
            elif 'via_m7m8' in m_path:
                via_m7m8 = np.genfromtxt(m_path, delimiter = ',')
                via_m7m8 = via_m7m8/via_m7m8.max()
                via_m7m8 = torch.tensor(resize(via_m7m8, [512,512])).float().unsqueeze(0)
            elif 'via_m8m9' in m_path:
                via_m8m9 = np.genfromtxt(m_path, delimiter = ',')
                via_m8m9 = via_m8m9/via_m8m9.max()
                via_m8m9 = torch.tensor(resize(via_m8m9, [512,512])).float().unsqueeze(0)
            # elif 'current_source' in m_path:
            #     current_source = np.genfromtxt(m_path, delimiter = ',')
            #     current_source = current_source/current_source.max()
            #     current_source = torch.tensor(resize(current_source, [512,512])).float().unsqueeze(0)
                    
        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7, 
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9, 
                             ir_drop], dim=0)

        return data    
    
    def __len__(self):
        return len(self.maps)
    
class load_real(data.Dataset):
    def __init__(self, folder_path, mode='train', testcase = []):
        self.folder_path = folder_path
        self.folder_list = os.listdir(folder_path)
        self.folder_list = sorted(self.folder_list)
        print(self.folder_list)
        if mode == 'train':
            self.folder_list = [f for f in self.folder_list if f not in testcase]
        else:
            self.folder_list = [f for f in self.folder_list if f in testcase]
        
    def __len__(self):
        return len(self.folder_list)
    
    def __getitem__(self, index):
        folder_name = self.folder_list[index]
        folder_dir = os.path.join(self.folder_path, folder_name)
        
        current = np.genfromtxt(os.path.join(folder_dir,'current_map.csv'), delimiter = ',')
        current = current/current.max() #(current-current.mean())/current.std()
        current = torch.tensor(resize(current, [512,512])).float().unsqueeze(0)
        
        dist = np.genfromtxt(os.path.join(folder_dir,'eff_dist_map.csv'), delimiter = ',')
        dist = dist/dist.max()
        dist = torch.tensor(resize(dist, [512,512])).float().unsqueeze(0)
        
        pdn = np.genfromtxt(os.path.join(folder_dir,'pdn_density.csv'), delimiter = ',')
        pdn = pdn/pdn.max()
        pdn = torch.tensor(resize(pdn, [512,512])).float().unsqueeze(0)
        
        resistance_m1 = np.genfromtxt(os.path.join(folder_dir,'resistance_m1.csv'), delimiter = ',')
        resistance_m1 = resistance_m1/resistance_m1.max()
        resistance_m1 = torch.tensor(resize(resistance_m1, [512,512])).float().unsqueeze(0)
        
        resistance_m4 = np.genfromtxt(os.path.join(folder_dir,'resistance_m4.csv'), delimiter = ',')
        resistance_m4 = resistance_m4/resistance_m4.max()
        resistance_m4 = torch.tensor(resize(resistance_m4, [512,512])).float().unsqueeze(0)
        
        resistance_m7 = np.genfromtxt(os.path.join(folder_dir,'resistance_m7.csv'), delimiter = ',')
        resistance_m7 = resistance_m7/resistance_m7.max()
        resistance_m7 = torch.tensor(resize(resistance_m7, [512,512])).float().unsqueeze(0)
        
        resistance_m8 = np.genfromtxt(os.path.join(folder_dir,'resistance_m8.csv'), delimiter = ',')
        resistance_m8 = resistance_m8/resistance_m8.max()
        resistance_m8 = torch.tensor(resize(resistance_m8, [512,512])).float().unsqueeze(0)
        
        resistance_m9 = np.genfromtxt(os.path.join(folder_dir,'resistance_m9.csv'), delimiter = ',')
        resistance_m9 = resistance_m9/resistance_m9.max()
        resistance_m9 = torch.tensor(resize(resistance_m9, [512,512])).float().unsqueeze(0)
        
        via_m1m4 = np.genfromtxt(os.path.join(folder_dir,'via_m1m4.csv'), delimiter = ',')
        via_m1m4 = via_m1m4/via_m1m4.max()
        via_m1m4 = torch.tensor(resize(via_m1m4, [512,512])).float().unsqueeze(0)
        
        via_m4m7 = np.genfromtxt(os.path.join(folder_dir,'via_m4m7.csv'), delimiter = ',')
        via_m4m7 = via_m4m7/via_m4m7.max()
        via_m4m7 = torch.tensor(resize(via_m4m7, [512,512])).float().unsqueeze(0)
        
        via_m7m8 = np.genfromtxt(os.path.join(folder_dir,'via_m7m8.csv'), delimiter = ',')
        via_m7m8 = via_m7m8/via_m7m8.max()
        via_m7m8 = torch.tensor(resize(via_m7m8, [512,512])).float().unsqueeze(0)
        
        via_m8m9 = np.genfromtxt(os.path.join(folder_dir,'via_m8m9.csv'), delimiter = ',')
        via_m8m9 = via_m8m9/via_m8m9.max()
        via_m8m9 = torch.tensor(resize(via_m8m9, [512,512])).float().unsqueeze(0)
        
        # current_source = np.genfromtxt(os.path.join(folder_dir,'current_source.csv'), delimiter = ',')
        # current_source = current_source/current_source.max()
        # current_source = torch.tensor(resize(current_source, [512,512])).float().unsqueeze(0)
        
        
        ir_drop = np.genfromtxt(os.path.join(folder_dir,'ir_drop_map.csv'), delimiter = ',')
        ir_drop = torch.tensor(resize(ir_drop, [512,512])).float().unsqueeze(0)
        
        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7, 
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9, 
                             ir_drop], dim=0)
        
        return data
        
        
class load_real_original_size(data.Dataset):
    def __init__(self, folder_path, mode='train', testcase = [], print_name = False):
        self.folder_path = folder_path
        self.folder_list = os.listdir(folder_path)
        self.folder_list = sorted(self.folder_list)
        if mode == 'train':
            self.folder_list = [f for f in self.folder_list if f not in testcase]
        else:
            self.folder_list = [f for f in self.folder_list if f in testcase]
        if print_name:
            print(self.folder_list)
        
    def __len__(self):
        return len(self.folder_list)
    
    def __folderlist__(self):
        return self.folder_list
    
    def __getitem__(self, index):
        folder_name = self.folder_list[index]
        folder_dir = os.path.join(self.folder_path, folder_name)
        
        current = np.genfromtxt(os.path.join(folder_dir,'current_map.csv'), delimiter = ',')
        current = current/current.max()
        shape = current.shape
        current = torch.tensor(current).float().unsqueeze(0)
        
        
        dist = np.genfromtxt(os.path.join(folder_dir,'eff_dist_map.csv'), delimiter = ',')
        dist = dist/dist.max()
        dist = torch.tensor(dist).float().unsqueeze(0)
        
        pdn = np.genfromtxt(os.path.join(folder_dir,'pdn_density.csv'), delimiter = ',')
        pdn = pdn/pdn.max()
        pdn = torch.tensor(pdn).float().unsqueeze(0)
        
        resistance_m1 = np.genfromtxt(os.path.join(folder_dir,'resistance_m1.csv'), delimiter = ',')
        resistance_m1 = resistance_m1/resistance_m1.max()        
        resistance_m1 = torch.tensor(resize(resistance_m1, shape)).float().unsqueeze(0)
        
        resistance_m4 = np.genfromtxt(os.path.join(folder_dir,'resistance_m4.csv'), delimiter = ',')
        resistance_m4 = resistance_m4/resistance_m4.max()
        resistance_m4 = torch.tensor(resize(resistance_m4, shape)).float().unsqueeze(0)
        
        resistance_m7 = np.genfromtxt(os.path.join(folder_dir,'resistance_m7.csv'), delimiter = ',')
        resistance_m7 = resistance_m7/resistance_m7.max()
        resistance_m7 = torch.tensor(resize(resistance_m7, shape)).float().unsqueeze(0)
        
        resistance_m8 = np.genfromtxt(os.path.join(folder_dir,'resistance_m8.csv'), delimiter = ',')
        resistance_m8 = resistance_m8/resistance_m8.max()
        resistance_m8 = torch.tensor(resize(resistance_m8, shape)).float().unsqueeze(0)
        
        resistance_m9 = np.genfromtxt(os.path.join(folder_dir,'resistance_m9.csv'), delimiter = ',')
        resistance_m9 = resistance_m9/resistance_m9.max()
        resistance_m9 = torch.tensor(resize(resistance_m9, shape)).float().unsqueeze(0)
        
        via_m1m4 = np.genfromtxt(os.path.join(folder_dir,'via_m1m4.csv'), delimiter = ',')
        via_m1m4 = via_m1m4/via_m1m4.max()
        via_m1m4 = torch.tensor(resize(via_m1m4, shape)).float().unsqueeze(0)
        
        via_m4m7 = np.genfromtxt(os.path.join(folder_dir,'via_m4m7.csv'), delimiter = ',')
        via_m4m7 = via_m4m7/via_m4m7.max()
        via_m4m7 = torch.tensor(resize(via_m4m7, shape)).float().unsqueeze(0)
        
        via_m7m8 = np.genfromtxt(os.path.join(folder_dir,'via_m7m8.csv'), delimiter = ',')
        via_m7m8 = via_m7m8/via_m7m8.max()
        via_m7m8 = torch.tensor(resize(via_m7m8, shape)).float().unsqueeze(0)
        
        via_m8m9 = np.genfromtxt(os.path.join(folder_dir,'via_m8m9.csv'), delimiter = ',')
        via_m8m9 = via_m8m9/via_m8m9.max()
        via_m8m9 = torch.tensor(resize(via_m8m9, shape)).float().unsqueeze(0)
        
        # current_source = np.genfromtxt(os.path.join(folder_dir,'current_source.csv'), delimiter = ',')
        # current_source = current_source/current_source.max()
        # current_source = torch.tensor(resize(current_source, shape)).float().unsqueeze(0)

        ir_drop = np.genfromtxt(os.path.join(folder_dir,'ir_drop_map.csv'), delimiter = ',')
        ir_drop = torch.tensor(ir_drop).float().unsqueeze(0)
            
        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7, 
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9, 
                             ir_drop], dim=0)
        
        return data
        
        