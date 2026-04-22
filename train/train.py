import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from heatmap_dataset import HeatmapLandmarkDataset
from heatmap_ustransunet import get_deit_heatmap_model, HeatmapLoss, extract_coordinates, euclidean_distance
from logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="train args")
    parser.add_argument('--train_csv', type=str, default='train.csv', help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='val.csv', help='Path to validation CSV file')
    parser.add_argument('--train_dir', type=str, default='train', help='Directory with training images')
    parser.add_argument('--val_dir', type=str, default='val', help='Directory with validation images')
    parser.add_argument('--heatmap_size', type=int, default=64, help='Size of the heatmap')
    parser.add_argument('--sigma', type=float, default=4.0, help='Standard deviation for Gaussian heatmap')
    parser.add_argument('--num_keypoints', type=int, default=3, help='Number of keypoints')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr_step', type=int, default=50, help='Learning rate decay step (decay every n epochs)')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--pretrained', type=str, default='model.pth.tar', help='Path to pretrained model weights')
    parser.add_argument('--save_dir', type=str, default='logs', help='Directory to save results')
    parser.add_argument('--eval_interval', type=int, default=5, help='Interval to evaluate on validation')
    args = parser.parse_args()
    return args


def interpolate_pos_embed(pretrained_pos_embed, target_size):
    cls_token = pretrained_pos_embed[:, 0:1, :]
    patch_pos_embed = pretrained_pos_embed[:, 1:, :]
    old_num_patches = patch_pos_embed.shape[1]
    new_num_patches = target_size - 1
    old_size = int(old_num_patches ** 0.5)
    new_size = int(new_num_patches ** 0.5)
    patch_pos_embed = patch_pos_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
    new_pos_embed = torch.cat([cls_token, patch_pos_embed], dim=1)
    return new_pos_embed


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_coord_distance = 0.0
    with tqdm(train_loader, desc=f"Training Epoch {epoch}") as pbar:
        for i, (images, heatmaps, landmarks) in enumerate(pbar):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            landmarks = landmarks.to(device)
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_coords = extract_coordinates(pred_heatmaps)
                coord_distance = torch.mean(euclidean_distance(pred_coords, landmarks))
            batch_loss = loss.item()
            total_loss += batch_loss
            total_coord_distance += coord_distance.item()
    avg_loss = total_loss / len(train_loader)
    avg_coord_distance = total_coord_distance / len(train_loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    writer.add_scalar('Coord_Distance/train_epoch', avg_coord_distance, epoch)
    return avg_loss, avg_coord_distance


@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    total_coord_distance = 0.0
    with tqdm(val_loader, desc=f"Validation Epoch {epoch}") as pbar:
        for i, (images, heatmaps, landmarks) in enumerate(pbar):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            landmarks = landmarks.to(device)
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, heatmaps)
            pred_coords = extract_coordinates(pred_heatmaps)
            coord_distance = torch.mean(euclidean_distance(pred_coords, landmarks))
            total_loss += loss.item()
            total_coord_distance += coord_distance.item()
    avg_loss = total_loss / len(val_loader)
    avg_coord_distance = total_coord_distance / len(val_loader)
    writer.add_scalar('Loss/val_epoch', avg_loss, epoch)
    writer.add_scalar('Coord_Distance/val_epoch', avg_coord_distance, epoch)
    return avg_loss, avg_coord_distance


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    with open(os.path.join(args.save_dir, 'config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    logger = setup_logger(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    train_dataset = HeatmapLandmarkDataset(csv_file=args.train_csv, img_dir=args.train_dir, transform=train_transform, train=True, heatmap_size=args.heatmap_size, sigma=args.sigma)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = HeatmapLandmarkDataset(csv_file=args.val_csv, img_dir=args.val_dir, transform=None, train=False, heatmap_size=args.heatmap_size, sigma=args.sigma)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = get_deit_heatmap_model(num_keypoints=args.num_keypoints, heatmap_size=args.heatmap_size)
    model = model.to(device)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            logger.info(f"Loading pretrained model weights from '{args.pretrained}'")
            state_dict = torch.load(args.pretrained, map_location=device)
            state_dict = state_dict['model']
            new_state_dict = {
                k.replace('model.', 'vit_bottleneck.'): v
                for k, v in state_dict.items()
                if k.startswith('model.')
            }
            if 'vit_bottleneck.pos_embed' in new_state_dict:
                new_state_dict['vit_bottleneck.pos_embed'] = interpolate_pos_embed(
                    new_state_dict['vit_bottleneck.pos_embed'], target_size=model.vit_bottleneck.pos_embed.shape[1]
                )
            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {args.pretrained}")
        else:
            logger.warning(f"No pretrained model found at '{args.pretrained}'")
    logger.info(model)
    criterion = HeatmapLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    start_epoch = 1
    best_val_coord_distance = float('inf')
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_coord_distance = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        val_loss, val_coord_distance = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        scheduler.step()
        logger.info(f"Epoch {epoch}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Coord Distance: {512 * train_coord_distance:.4f} - "
                   f"Val Loss: {val_loss:.4f}, Val Coord Distance: {512 * val_coord_distance:.4f} - "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on validation coordinate distance
        if val_coord_distance < best_val_coord_distance:
            best_val_coord_distance = val_coord_distance
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', f'best_val_epoch_{epoch}_{val_coord_distance*512:.3f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_coord_distance': best_val_coord_distance
            }, checkpoint_path)
            logger.info(f"Saved best validation model to {checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*80)
    logger.info(f"Results saved to: {args.save_dir}")
    logger.info(f"Best validation coordinate distance: {best_val_coord_distance*512:.3f} pixels")
    logger.info("="*80)


if __name__ == "__main__":
    args = parse_args()
    main(args)