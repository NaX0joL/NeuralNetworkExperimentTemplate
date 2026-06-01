import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F

from .config import SimpleCNNConfig



class SimpleCNN2(nn.Module):
    def __init__(self, args, config:SimpleCNNConfig):
        super().__init__()
        
        self.config = config
        
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.proposed_arch1()
        return
    
    def proposed_arch1(self):
        # hyperparam manual init
        self.filter_number = [
            64, 128, 256, 512,
        ]
        self.conv_kernel_size = [
            # 32, 16, 8, 5,
            3, 3, 3, 3,
        ]
        self.linear_dim = [
            1000, 1000
        ]
        
        # encoder
        self.conv = nn.ModuleList([
            nn.Conv1d(1, self.filter_number[0], kernel_size=self.conv_kernel_size[0], padding=1),
            nn.Conv1d(self.filter_number[0], self.filter_number[1], kernel_size=self.conv_kernel_size[1], padding=1),
            nn.Conv1d(self.filter_number[1], self.filter_number[2], kernel_size=self.conv_kernel_size[2], padding=1),
            nn.Conv1d(self.filter_number[2], self.filter_number[3], kernel_size=self.conv_kernel_size[3], padding=1),
        ])
        self.pooling = nn.ModuleList([
            nn.MaxPool1d(2, 2, return_indices=True),
            nn.MaxPool1d(2, 2, return_indices=True),
            nn.MaxPool1d(2, 2, return_indices=True),
            nn.MaxPool1d(2, 2, return_indices=True),
        ])
        
        self.flatten = nn.Flatten()
        
        # the linear connecting the encoder/latent and latent/decoder
        # is dynamically defined based on the input dim
        self.connector1 = None
        self.connector2 = None
        
        # latent space
        self.latent = nn.Sequential(
            nn.Linear(self.linear_dim[0], self.linear_dim[1]),
            nn.ReLU(),
        )
        
        self.unflatten = None
        
        # decoder
        self.deconv = nn.ModuleList([
            nn.ConvTranspose1d(self.filter_number[3], self.filter_number[2], kernel_size=self.conv_kernel_size[3], padding=1),
            nn.ConvTranspose1d(self.filter_number[2], self.filter_number[1], kernel_size=self.conv_kernel_size[2], padding=1),
            nn.ConvTranspose1d(self.filter_number[1], self.filter_number[0], kernel_size=self.conv_kernel_size[1], padding=1),
            nn.ConvTranspose1d(self.filter_number[0], 1, kernel_size=self.conv_kernel_size[0], padding=1),
        ])
        self.unpool = nn.ModuleList([
            nn.MaxUnpool1d(2, 2),
            nn.MaxUnpool1d(2, 2),
            nn.MaxUnpool1d(2, 2),
            nn.MaxUnpool1d(2, 2),
        ])
        
        return
    
    def proposed_arch2(self):
        pass
    
    def forward(self, x:Tensor):
        x = x.permute(0, 2, 1)      # [bs, feature, seq_len]
        
        # encoder
        pooling_indices = []
        sizes = []
        for conv, pooling in zip(self.conv, self.pooling):
            x = F.relu(conv(x))
            sizes.append(x.size())
            x, indices = pooling(x)
            pooling_indices.append(indices)
        
        # dynamically define connector between latent and encoder/decoder
        if self.connector1 is None:
            bs, channels, seq_len = x.shape
            flattened_size = channels*seq_len
            self.connector1 = nn.Linear(flattened_size, self.linear_dim[0]).to(x.device)
            self.connector2 = nn.Linear(self.linear_dim[-1], flattened_size).to(x.device)
            self.unflatten = nn.Unflatten(1, (channels, seq_len)).to(x.device)
        
        # latent
        x = self.flatten(x)
        x = F.relu(self.connector1(x))
        x = self.latent(x)
        x = F.relu(self.connector2(x))
        x = self.unflatten(x)
        
        # decoder
        for index, (unpool, deconv, indices, size) in enumerate(zip(self.unpool, self.deconv, 
                                                                    reversed(pooling_indices), 
                                                                    reversed(sizes))):
            x = unpool(x, indices, output_size=size)
            x = deconv(x)
            if index < len(self.unpool) - 1:
                x = F.relu(x)
        
        # x = F.sigmoid(x)
        
        x = x.permute(0, 2, 1)      # [bs, seq_len, feature]
        return x



# def train_cnn(model_name, dataset_name):
#     model = SimpleCNN()
#     model = model.to(model.device)
    
#     args = get_default_args()
    
#     args.for_training = True
#     args.dataset_shuffle = True
#     _, train_dataloader = get_dataset(args)
#     args.for_training = False
#     args.dataset_shuffle = False
#     _, test_dataloader = get_dataset(args)
    
#     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#     criterion = nn.HuberLoss(reduction=args.huber_reduction, delta=args.huber_delta)
    
#     train_loss_log = []
#     val_loss_log = []
    
#     for epoch in range(args.train_epochs):
#         time_start = time.time()
        
#         model.train()
#         for batch_data in train_dataloader:
#             batch_value = batch_data['value'].float().to(model.device)
#             batch_gt = batch_data['ground_truth'].float().to(model.device)
            
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 output = model(batch_value)
                
#                 loss = criterion(output, batch_gt)
                
#                 loss.backward()
#                 if args.use_grad_clip: nn.utils.clip_grad_norm_(model.parameters(), args.clipping_value)
#                 optimizer.step()
#         train_loss = loss
        
#         model.eval()
#         val_loss = 0
#         for batch_data in test_dataloader:
#             batch_value = batch_data['value'].float().to(model.device)
#             batch_gt = batch_data['ground_truth'].float().to(model.device)
            
#             with torch.no_grad():
#                 output = model(batch_value)
                
#                 loss = criterion(output, batch_gt)
#                 val_loss += loss
        
#         time_end = time.time()
                
#         one_epoch_time = time_end - time_start
        
#         train_loss /= len(train_dataloader)
#         val_loss /= len(test_dataloader)
        
#         train_loss_log.append(train_loss)
#         val_loss_log.append(val_loss)
        
#         print(f"===== EPOCH-{epoch+1} =====")
#         print(f"average loss:")
#         print(f"  train : {train_loss:.6f}")
#         print(f"  test  : {val_loss:.6f}")
#         print(f"time cost: {one_epoch_time:.3f}")
#         print()
    
#     model_path = f'savefolder/modelPack/model.pth'
#     torch.save(model.state_dict(), model_path)
    
#     return model