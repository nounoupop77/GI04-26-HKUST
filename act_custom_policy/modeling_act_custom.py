import torch
import torch.nn as nn
import torchvision.models as models

class ACTCustomPolicy(nn.Module):
    def __init__(self, state_dim=6, action_dim=6, chunk_size=100, hidden_dim=512, latent_dim=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # 1. Vision Backbone (ResNet18)
        resnet_front = models.resnet18(weights="DEFAULT")
        resnet_side = models.resnet18(weights="DEFAULT")
        self.encoder_front = nn.Sequential(*list(resnet_front.children())[:-1])
        self.encoder_side = nn.Sequential(*list(resnet_side.children())[:-1])
        
        vision_feat_dim = 512 + 512  # Front + Side

        # 2. CVAE Encoder (Training Only)
        # Takes vision features and ground-truth action chunk
        self.vae_encoder = nn.Sequential(
            nn.Linear(vision_feat_dim + (action_dim * chunk_size), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.vae_mu = nn.Linear(hidden_dim, latent_dim)
        self.vae_logvar = nn.Linear(hidden_dim, latent_dim)

        # 3. Transformer Decoder
        self.input_proj_vision = nn.Linear(vision_feat_dim, hidden_dim)
        self.input_proj_state = nn.Linear(state_dim, hidden_dim)
        self.input_proj_latent = nn.Linear(latent_dim, hidden_dim)
        
        # Learnable queries for the action sequence
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, hidden_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, img_front, img_side, state, actions=None):
        # Extract Vision Features
        feat_front = self.encoder_front(img_front).flatten(1)
        feat_side = self.encoder_side(img_side).flatten(1)
        vision_features = torch.cat([feat_front, feat_side], dim=1)

        # Handle Latent Vector (z)
        if actions is not None:
            # Training: Encode ground-truth actions
            actions_flat = actions.flatten(1)
            vae_input = torch.cat([vision_features, actions_flat], dim=1)
            vae_hidden = self.vae_encoder(vae_input)
            mu = self.vae_mu(vae_hidden)
            logvar = self.vae_logvar(vae_hidden)
            
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # Inference: z is zero
            mu, logvar = None, None
            z = torch.zeros((img_front.shape[0], self.latent_dim), device=img_front.device)

        # Prepare Transformer Memory
        vision_token = self.input_proj_vision(vision_features).unsqueeze(1)
        state_token = self.input_proj_state(state).unsqueeze(1)
        latent_token = self.input_proj_latent(z).unsqueeze(1)
        memory = torch.cat([vision_token, state_token, latent_token], dim=1)

        # Transformer Forward
        queries = self.action_queries.expand(img_front.shape[0], -1, -1)
        transformer_out = self.transformer(tgt=queries, memory=memory)
        predicted_actions = self.action_head(transformer_out)

        return predicted_actions, mu, logvar