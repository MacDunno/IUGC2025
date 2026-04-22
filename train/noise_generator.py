import pandas as pd
import numpy as np

def parse_coordinates(coord_str):
    coord_str = coord_str.strip('()')
    x, y = map(float, coord_str.split(', '))
    return x, y

def add_gaussian_noise(x, y, sigma=2.0):
    noise_x = np.random.normal(0, sigma)
    noise_y = np.random.normal(0, sigma)
    return x + noise_x, y + noise_y

def perturb_coordinates(coord_str, sigma=2.0):
    x, y = parse_coordinates(coord_str)
    x_new, y_new = add_gaussian_noise(x, y, sigma)
    x_new = int(round(x_new))
    y_new = int(round(y_new))
    return f"({x_new}, {y_new})"

def calculate_aop(ps1, ps2, fh1):
    ps1_x, ps1_y = parse_coordinates(ps1)
    ps2_x, ps2_y = parse_coordinates(ps2)
    fh1_x, fh1_y = parse_coordinates(fh1)
    vec1 = np.array([ps2_x - ps1_x, ps2_y - ps1_y])
    vec2 = np.array([fh1_x - ps1_x, fh1_y - ps1_y])
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0) 
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle

def perturb_labels(csv_file, output_file, sigma=2.0):
    df = pd.read_csv(csv_file)
    coord_columns = ['PS1', 'PS2', 'FH1']
    for col in coord_columns:
        df[col] = df[col].apply(lambda x: perturb_coordinates(x, sigma))
    df['AOP'] = df.apply(lambda row: calculate_aop(row['PS1'], row['PS2'], row['FH1']), axis=1)
    df.to_csv(output_file, index=False)
    print(f"save in: {output_file}")