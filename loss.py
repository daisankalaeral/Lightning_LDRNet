import torch
import torch.nn as nn
import torch.nn.functional as F
import configs

def line_loss(line):
        line_x = line[:, 0::2]
        line_y = line[:, 1::2]
        
        x_diff = line_x[:, 1:] - line_x[:, 0:-1]
        y_diff = line_y[:, 1:] - line_y[:, 0:-1]
        
        x_diff_start = x_diff[:, 1:]
        x_diff_end = x_diff[:, 0:-1]
        y_diff_start = y_diff[:, 1:]
        y_diff_end = y_diff[:, 0:-1]
        
        similarity = (x_diff_start * x_diff_end + y_diff_start * y_diff_end) / (torch.sqrt(torch.square(x_diff_start) + torch.square(y_diff_start) + 1e-10) * torch.sqrt(torch.square(x_diff_end) + torch.square(y_diff_end) + 1e-10))
        
        slop_loss = torch.mean(1 - similarity, dim=1)
        
        x_diff_loss = torch.mean(torch.square(x_diff[:, 1:] - x_diff[:, 0:-1]), dim=1)
        y_diff_loss = torch.mean(torch.square(y_diff[:, 1:] - y_diff[:, 0:-1]), dim=1)
        
        return slop_loss, x_diff_loss + y_diff_loss

def calculate_line_loss(y_pred):
    
    total_slop_loss = 0
    total_diff_loss = 0

    for index in range(4):
        line = y_pred[:, index * 2:index * 2 + 2]
        for coord_index in range(configs.size_per_border):
            line = torch.concat(
                [line, y_pred[:, 8 + coord_index * 8 + index * 2:8 + coord_index * 8 + index * 2 + 2]], axis=1)
        line = torch.concat([line, y_pred[:, (index * 2 + 2) % 8:(index * 2 + 2 + 2) % 8]], axis=1)
        cur_slop_loss, cur_diff_loss = line_loss(line)
        total_slop_loss += cur_slop_loss
        total_diff_loss += cur_diff_loss
    return configs.beta * total_slop_loss + configs.gamma * total_diff_loss

def calculate_total_loss(corner_coords_true, corner_coords_pred, border_coords_pred, cls_true, cls_pred):
    coord_start = corner_coords_true[:, 0:8]
    coord_end = torch.concat([corner_coords_true[:, 2:8], corner_coords_true[:, 0:2]], axis=1)
    coord_increment = (coord_end - coord_start) / (configs.size_per_border + 1)
    new_coord = coord_start + coord_increment

    for index in range(1, configs.size_per_border):
        new_coord = torch.concat([new_coord, coord_start + (index + 1) * coord_increment], axis=1)
    
    y = torch.concat([coord_start, new_coord], axis = 1)
    y_pred = torch.concat([corner_coords_pred, border_coords_pred], axis = 1)
    
    reg_loss = torch.nn.functional.mse_loss(y_pred, y, reduction = "none")
    mean_reg_loss = torch.mean(reg_loss, dim=1)
    
    line_loss = calculate_line_loss(y_pred)

    cls_loss = F.cross_entropy(cls_pred, cls_true, reduction = "none")

    
    total_loss = torch.mean(configs.reg_ratio * mean_reg_loss + line_loss + cls_loss)
    
    return total_loss