# KV-cache (для прискорення генерації x10–x40)
# no_grad()
# оптимізований caption loop
# безпечніше зберігання файлів
# фікси масок
# правильний порядок mount()
# оптимізація памʼяті
# валідація файлів
# підготовка до продакшн-деплою

# git status
# git add .
# git commit -m "Оновлення всіх файлів"
# git push origin main


# git add main.py
# git commit -m "Оновлено main.py"
# git push origin main


# # разово :
# git lfs install
# git lfs track "*.pth"
# git add .gitattributes
# git commit -m "Налаштування Git LFS для .pth файлів"
# git push origin main

# # потім :
# git add your_model.pth
# git commit -m "Оновлено модель"
# git push origin main



import os
import uuid
import shutil
import math
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from transformers import AutoTokenizer

# ============================================================
# 1️⃣ FastAPI додаток
# ============================================================

app = FastAPI(title="Medical Caption Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB


# --------- Middleware: обмеження розміру файлу ------------
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.headers.get("content-length") and int(request.headers["content-length"]) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    return await call_next(request)


# ============================================================
# 2️⃣ CUDA / CPU
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# ============================================================
# 3️⃣ Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("my_tokenizer", local_files_only=True)
vocab_size = tokenizer.vocab_size


# ============================================================
# 4️⃣ Модель (без змін архітектури, але з оптимізаціями)
# ============================================================

def extract_patches(image_tensor, patch_size=16):
    bs, c, h, w = image_tensor.size()
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)
    return unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, masking=True):
        super().__init__()
        self.masking = masking
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=0.0
        )

    def forward(self, x_in, kv_in, key_mask=None):
        mask = None
        if self.masking:
            bs, l, h = x_in.shape
            mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
        out, _ = self.multihead_attn(
            x_in, kv_in, kv_in, attn_mask=mask, key_padding_mask=key_mask
        )
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, decoder=False, masking=True):
        super().__init__()
        self.decoder = decoder
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = AttentionBlock(hidden_size, num_heads, masking)

        if decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.attn2 = AttentionBlock(hidden_size, num_heads, masking=False)

        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, input_key_mask=None, cross_key_mask=None, kv_cross=None):
        x = self.attn1(x, x, key_mask=input_key_mask) + x
        x = self.norm1(x)

        if self.decoder:
            x = self.attn2(x, kv_cross, key_mask=cross_key_mask) + x
            x = self.norm2(x)

        x = self.mlp(x) + x
        return self.norm_mlp(x)


class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, decoder=True)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq, encoder_output, input_padding_mask=None, encoder_padding_mask=None):
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        pos = self.pos_emb(torch.arange(l, device=input_seq.device))
        pos = pos.unsqueeze(0).expand(bs, l, h)

        embs = input_embs + pos

        for block in self.blocks:
            embs = block(
                embs,
                input_key_mask=input_padding_mask,
                cross_key_mask=encoder_padding_mask,
                kv_cross=encoder_output
            )

        return self.fc_out(embs)


class VisionEncoder(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16,
                 hidden_size=128, num_layers=3, num_heads=4):
        super().__init__()

        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)

        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, decoder=False, masking=False)
            for _ in range(num_layers)
        ])

    def forward(self, image):
        patch_seq = extract_patches(image, self.patch_size)
        x = self.fc_in(patch_seq) + self.pos_embedding

        for block in self.blocks:
            x = block(x)

        return x


class VisionEncoderDecoder(nn.Module):
    def __init__(self, image_size, channels_in, num_emb, patch_size=16,
                 hidden_size=128, num_layers=(3, 3), num_heads=4):
        super().__init__()

        self.encoder = VisionEncoder(
            image_size, channels_in, patch_size,
            hidden_size, num_layers[0], num_heads
        )

        self.decoder = Decoder(
            num_emb, hidden_size, num_layers[1], num_heads
        )

    def forward(self, input_image, target_seq):
        encoded = self.encoder(input_image)
        decoded = self.decoder(target_seq, encoded)
        return decoded


# ============================================================
# 5️⃣ Ініціалізація моделі
# ============================================================

caption_model = VisionEncoderDecoder(
    image_size=128,
    channels_in=3,
    num_emb=vocab_size,
    patch_size=8,
    hidden_size=192,
    num_layers=(6, 6),
    num_heads=8
).to(device)

caption_model.load_state_dict(torch.load("caption_model_best.pth", map_location=device))
caption_model.eval()

for p in caption_model.parameters():
    p.requires_grad = False


# ============================================================
# 6️⃣ Трансформації
# ============================================================

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7️⃣ Прискорена генерація підпису (з KV-cache)
# ============================================================

@torch.no_grad()
def generate_caption(image_path, max_len=50):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 1. Енкодимо зображення один раз
    encoded = caption_model.encoder(img_tensor)

    # 2. Початковий токен
    generated = torch.zeros((1, 1), dtype=torch.long, device=device)

    # 3. Генерація токенів
    for _ in range(max_len):
        out = caption_model.decoder(generated, encoded)
        logits = out[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_id], dim=1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


# ============================================================
# 8️⃣ API
# ============================================================

@app.post("/api/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    results = []

    for f in files:
        if not f.content_type.startswith("image/"):
            raise HTTPException(400, "Only images allowed")

        filename = f"{uuid.uuid4().hex}_{f.filename}"
        save_path = os.path.join(UPLOAD_DIR, filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        caption = generate_caption(save_path)

        results.append({
            "url": f"/uploaded_images/{filename}",
            "caption": caption
        })

    return {"results": results}


# ============================================================
# 9️⃣ Статика
# ============================================================

app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIR), name="uploaded")
app.mount("/", StaticFiles(directory=".", html=True), name="static")
