import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

from collections import deque
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import onnxruntime
import torch.onnx
import onnx
from torchsummary import summary

# import tensorflowjs as tfjs
# import tensorflow as tf
# from onnx_tf.backend import prepare

# from utils import save_model, load_model, augmented_train_set, augmented_val_set

# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

# device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")


class Encoder(nn.Module):
    def __init__(self, img_size, label_size, latent_size, hidden_size):
        super(Encoder, self).__init__()
        self.img_size = img_size  # (C, H, W)
        self.label_size = label_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        
        self.flat_img_size = img_size[0] * img_size[1] * img_size[2]
        self.fc_img_enc = nn.Linear(self.flat_img_size, self.hidden_size)
        self.fc_lbl_enc = nn.Linear(self.label_size, self.hidden_size)
        self.encoder = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.encoder2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_mu = nn.Linear(self.hidden_size, self.latent_size)
        self.fc_logstd = nn.Linear(self.hidden_size, self.latent_size)

    def forward(self, x, y):
        x = x.view(x.size(0), -1).float()
        if y.shape[-1] != self.label_size:
            y = F.one_hot(y, num_classes=self.label_size).float()
        y = y.float()
        x = F.relu(self.fc_img_enc(x))
        y = F.relu(self.fc_lbl_enc(y))
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.encoder(x))
        x = F.relu(self.encoder2(x))
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        return mu, logstd

    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor):
        std_dev = torch.exp(logstd * 0.5)
        eps = torch.randn_like(std_dev)
        z = mu + eps * std_dev
        return z

    def encode(self, x, y):
        mu, logstd = self.forward(x, y)
        z = self.reparametrize(mu, logstd)
        return z, mu, logstd


class Decoder(nn.Module):
    def __init__(self, img_size, label_size, latent_size, hidden_size):
        super(Decoder, self).__init__()
        self.img_size = img_size  # (C, H, W)
        self.label_size = label_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        
        self.fc_latent = nn.Linear(self.latent_size, self.hidden_size)
        self.fc_lbl_dec = nn.Linear(self.label_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.decoder2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_dec = nn.Linear(self.hidden_size, self.flat_img_size)

    @property
    def flat_img_size(self):
        return self.img_size[0] * self.img_size[1] * self.img_size[2]

    def forward(self, z, y):
        if y.shape[-1] != self.label_size:
            y = F.one_hot(y, num_classes=self.label_size).float()
        y = y.float()
        z = F.relu(self.fc_latent(z))
        y = F.relu(self.fc_lbl_dec(y))
        z = torch.cat((z, y), dim=1)
        z = F.relu(self.decoder(z))
        z = F.relu(self.decoder2(z))
        x = self.fc_dec(z)
        x = x.view(x.size(0), *self.img_size)
        x = torch.sigmoid(x)  # Apply sigmoid activation to get pixel values in [0, 1]
        return x

class CVAE(nn.Module):
    def __init__(self, img_size, label_size, latent_size, hidden_size):
        super(CVAE, self).__init__()
        self.img_size = img_size  # (C, H, W)
        self.label_size = label_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.encoder = Encoder(img_size, label_size, latent_size, hidden_size)
        self.decoder = Decoder(img_size, label_size, latent_size, hidden_size)
        
    def forward(self, x, y):
        z, mu, logstd = self.encoder.encode(x, y)
        x_recon = self.decoder(z, y)
        return x_recon, mu, logstd
    
    def encode_param(self, x, y):
        # compute mu and logstd of p(z|x)
        mu, logstd = self.encoder(x, y)
        return mu, logstd
    
    def reparamaterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        # compute latent z with reparameterization trick
        std_dev = torch.exp(logstd*0.5)
        eps = torch.randn_like(std_dev)
        z = mu + eps * std_dev
        return z
    
    def encode(self, x, y):
        # sample latent z from p(z|x)
        mu, logstd = self.encode_param(x, y)
        z = self.reparamaterize(mu, logstd)
        return z, mu, logstd
    
    def decode(self, z, y):
        recon_x = self.decoder(z, y)
        return recon_x
    
    @torch.no_grad()
    def sample_images(self, label, save=True, save_dir='./vae'):
        self.eval()
        n_samples = label.shape[0]
        samples  = self.decoder.decode(torch.randn(n_samples, self.latent_size).to(label.device), label)
        imgs = samples.view(n_samples, 1, 28, 28).clamp(0., 1.)
        if save:
            os.makedirs(save_dir, exist_ok=True)
            torchvision.utils.save_image(imgs, os.path.join(save_dir, 'sample.png'), nrow=int(np.sqrt(n_samples)))
        return imgs
    

en_drate_schedule = torch.linspace(0.0, 1.0, 5)
de_drate_schedule = torch.linspace(0.0, 1.0, 5)

label_dim = 11
img_dim = (1, 28, 28)
latent_dim = 10
hidden_dim = 400

for en_drate in en_drate_schedule:
    for de_drate in de_drate_schedule:
        # if en_drate != 0.0 or de_drate != 0.0:
        #     break
        print("Drop label rate: ", en_drate, de_drate)
        vae_model = CVAE(img_size=img_dim, label_size=label_dim, latent_size=latent_dim, hidden_size=hidden_dim)
        # encoder = Encoder(img_size=img_dim, label_size=label_dim, latent_size=latent_dim, hidden_size=hidden_dim)
        # decoder = Decoder(img_size=img_dim, label_size=label_dim, latent_size=latent_dim, hidden_size=hidden_dim)
        encoder = vae_model.encoder
        decoder = vae_model.decoder
        
        # load pytorch model
        drate = f"{en_drate}_{de_drate}"
        path = "vae_models"
        encoder_path = os.path.join(path, f"encoder_{drate}.pth")
        decoder_path = os.path.join(path, f"decoder_{drate}.pth")
        vae_path = os.path.join(path, f"vae_model_{drate}.pth")
        
        if os.path.exists(encoder_path):
            encoder.load_state_dict(torch.load(encoder_path))
            print(f"Encoder loaded from {encoder_path}")
        else:
            print(f"Encoder not found at {encoder_path}")
        if os.path.exists(decoder_path):
            decoder.load_state_dict(torch.load(decoder_path))
            print(f"Decoder loaded from {decoder_path}")
        else:
            print(f"Decoder not found at {decoder_path}")
        if os.path.exists(vae_path):
            vae_model.load_state_dict(torch.load(vae_path))
            print(f"VAE model loaded from {vae_path}")
        else:
            print(f"VAE model not found at {vae_path}")
            
        encoder.eval()
        decoder.eval()
        vae_model.eval()
        
        # transfer the pytorch model to onnx model, save to corresponding onnx_path
        # encoder_input_image = torch.randn(1, *img_dim, requires_grad=True)
        encoder_input_image = torch.tensor([[[[0.52978152], [1.29756618], [-1.49313235], [-0.325250328], [-0.175024539], [-0.495998204], [-0.754435062], [0.15361616], [-1.81400144], [-0.115719199], [1.41716516], [0.941630006], [-1.06519902], [1.87934506], [-1.96668661], [-1.14171851], [0.119981848], [-1.24266684], [-1.28622901], [-0.0614364743], [-2.67685366], [-0.447020084], [-1.25021851], [1.65726864], [0.760278463], [-1.7155807], [-0.991976142], [1.92281973]], [[2.39339638], [-0.101350896], [1.77789676], [-0.74490869], [-1.48927784], [-0.685805798], [-0.448516816], [0.4939785], [1.10475314], [0.681169748], [-0.993800104], [-0.97552073], [-0.293419212], [-1.40669274], [-0.161728844], [0.0232218839], [0.637529552], [1.39261353], [0.806089997], [-1.06899023], [0.991831303], [0.247324735], [0.223213762], [0.788718522], [-1.29331088], [1.94727528], [-0.216277674], [1.69447482]], [[-0.579246879], [-1.66595912], [-1.30400944], [-0.802754223], [2.79492807], [1.77872705], [1.7419728], [1.25579822], [-1.45545268], [0.0532407314], [-0.32850343], [1.497226], [-0.115911514], [1.20735645], [-0.266393095], [0.334779561], [-0.663978457], [0.635533452], [0.469547242], [1.18821168], [-0.265565574], [-0.219489723], [-0.504570484], [-0.854393601], [-0.250634342], [0.419984996], [-0.750806212], [0.285951257]], [[0.290956348], [1.33599806], [-0.701409459], [0.489192665], [0.430651069], [1.42943895], [-2.10058975], [0.83740139], [-1.04471731], [1.29939556], [0.770427883], [-0.154885426], [-1.11341667], [-0.603437901], [-0.163308367], [0.000985166174], [-0.469361693], [-1.34452605], [-0.55906111], [-0.268941134], [1.38992202], [-0.66778034], [-1.74330151], [-1.44316983], [2.63738894], [-0.799137294], [-0.308642536], [-0.838228583]], [[-0.68325454], [1.10245383], [-0.195056319], [-0.304377884], [0.905285537], [0.00509236148], [-0.768879473], [-1.87652624], [-0.455117494], [-0.770831108], [-0.743210852], [-1.85496247], [0.219523817], [-1.88796258], [1.07524073], [-0.329298884], [0.148738042], [-0.710405469], [-1.5150038], [-0.146043658], [-0.851990759], [-1.49973273], [-0.786279857], [0.0403265134], [1.42897046], [-0.718880773], [2.61648321], [-0.749360979]], [[-0.309289992], [-0.0134086553], [-1.29709399], [0.0784833655], [-0.399439007], [-0.861024261], [-1.68540251], [0.373737037], [0.313598931], [0.504555285], [-1.80477536], [-0.428129166], [0.0111501403], [-1.3977704], [1.99205625], [-1.69346905], [-0.501947045], [1.59947443], [-0.916510642], [0.436603814], [0.267884463], [-1.27563512], [0.219621941], [0.329889625], [-1.1391679], [0.302246779], [-0.493266582], [0.883019626]], [[1.22258699], [-0.385142416], [0.325477809], [-1.20009828], [0.849602938], [-0.959170282], [-0.339601666], [0.617169738], [-0.97654593], [0.721345246], [-0.60828042], [-0.198574021], [0.805746675], [0.231028497], [0.854488611], [-0.943584919], [-0.450032443], [-0.200994313], [0.965310872], [-0.197082654], [1.41860139], [0.294232786], [0.0471305698], [-0.361021101], [1.14019704], [0.390910238], [-0.841478944], [0.808287024]], [[0.838549614], [-0.300085276], [2.14852023], [-0.325648218], [0.135877445], [0.0418037139], [0.502180696], [0.199074373], [0.680936158], [1.80704951], [0.180117667], [1.27259421], [0.592056632], [-0.451722682], [0.00761420745], [-0.761384606], [-0.53442204], [-0.290352315], [0.544594586], [-0.405586988], [2.03607917], [-1.66662025], [-1.21207345], [1.14315796], [1.13909984], [-0.766859412], [-1.21060562], [0.4049097]], [[0.882464767], [-0.697545409], [-0.317726016], [-0.203293115], [-0.772418439], [-0.130983442], [-1.01066744], [-1.21696734], [0.582744539], [1.2177918], [-1.22186434], [0.378638238], [0.925021887], [1.63564765], [0.606254458], [-0.101062901], [0.440785021], [1.15174186], [-0.0511873886], [-1.48409033], [0.146241754], [0.988253057], [2.69414234], [0.209490523], [-0.287189275], [-1.75208056], [0.746245801], [0.773720205]], [[0.852891564], [1.11716533], [-0.355245143], [0.0517202988], [-1.09970689], [0.459307671], [0.361013681], [1.32153308], [1.70749879], [1.07377088], [0.943717837], [-1.01414871], [0.705844462], [-0.774983287], [-0.483256251], [0.532174349], [0.0413222052], [0.339211375], [0.0154348612], [1.39136112], [0.634996831], [-0.321478307], [0.609217465], [-0.378405452], [0.953945279], [1.12483156], [-0.29684642], [-0.0436683595]], [[-0.514095128], [-0.404735833], [0.977180183], [-0.0661486387], [-1.36559057], [-0.30856204], [0.347993463], [-1.60328519], [2.08975959], [-1.94785321], [-0.472469896], [0.662843406], [1.84507728], [0.316250563], [-1.08629036], [1.09883118], [-0.83786726], [0.31149137], [0.120448098], [-1.40346432], [-0.474275082], [-1.2758652], [0.0506750122], [1.37111664], [1.26400876], [-0.453857183], [-1.62683916], [0.163087979]], [[1.09994328], [-0.328142345], [-0.887789607], [-2.25087881], [1.61086178], [0.319800794], [-0.152970135], [-2.96965456], [-0.0803184882], [1.51932967], [0.00972654391], [-0.252620816], [0.115593188], [1.11383915], [-0.203672066], [-0.951841116], [0.421223521], [0.126144379], [-1.1673466], [-1.88700008], [-0.373569101], [-0.415556699], [-0.602660596], [0.722319722], [-1.53280354], [-0.0572097674], [-1.46127927], [-0.353044212]], [[1.32577884], [1.11552918], [0.277113259], [-0.852220297], [-0.855504036], [-0.317253888], [-1.893448], [1.20997572], [0.314760029], [-0.561803937], [1.4396559], [1.11562073], [-1.01626003], [-0.884613931], [-1.21190572], [0.124662437], [-0.430297911], [0.344685644], [-1.53419662], [0.350535363], [1.04741848], [-0.570138931], [2.33664036], [-0.180057943], [0.573199987], [0.378743231], [-0.267138362], [-1.10457563]], [[0.0298031159], [-0.436241478], [-0.680963695], [1.33923745], [-0.0367860831], [0.194382519], [-0.644204497], [-0.0925362036], [-0.156383723], [0.62883687], [0.11223489], [-0.900514901], [-0.00209153933], [-0.623031974], [-0.472518861], [0.0887990966], [-0.613561809], [-1.83418214], [-0.121298626], [0.532444775], [0.277359366], [0.21059148], [-0.215295747], [-0.120760329], [0.0485637486], [0.686445355], [-0.867977142], [-0.587273479]], [[1.2226752], [-1.1071595], [1.40749419], [1.21820247], [-0.619881332], [-0.831538498], [-0.429566264], [-1.12832296], [0.502011061], [0.390456021], [-0.859026253], [-1.8851347], [-0.118731201], [0.20639661], [1.5243566], [1.26164055], [0.215024889], [-1.4224242], [1.46122658], [-1.53821945], [1.02131319], [1.02540481], [0.121929273], [-0.22939238], [-0.482672185], [1.82653773], [-0.343526334], [1.61535704]], [[0.4381136], [2.2078433], [0.0843313634], [-0.370246768], [0.38862735], [-0.0780159608], [0.203844011], [0.291473418], [0.685183108], [-0.297449231], [-0.292842925], [-0.396157473], [0.71307385], [0.691050291], [-0.467532247], [-0.0807631165], [-0.254513651], [-0.272777081], [0.280674577], [-2.29755902], [0.68183893], [0.538542271], [-0.708210528], [0.382989675], [0.814090967], [-0.0447285064], [-1.17611122], [1.44839466]], [[0.205254257], [1.67788303], [0.905499995], [0.873145878], [-0.0438814238], [-0.966156483], [0.973832548], [1.01614749], [-1.26790333], [-1.60170341], [0.73015368], [0.295120776], [-2.5590291], [1.56977093], [0.276569754], [0.16327402], [0.112759262], [-1.27809858], [-0.674357533], [1.05823874], [-0.138651714], [-1.77873075], [-0.113320112], [-2.66919589], [0.302522957], [0.442367971], [0.398958504], [-0.443850547]], [[0.541083336], [0.356290817], [-0.405497313], [-0.0965341032], [0.363623112], [0.61361748], [-0.497522891], [-0.764858067], [-1.31718194], [0.629485011], [-0.274076998], [0.235950455], [1.81189692], [1.0428884], [-0.428426296], [-0.881473541], [-0.22943598], [-0.533631384], [1.07039618], [-0.185960144], [0.229182571], [-0.702786624], [0.654496253], [0.52522397], [1.8463552], [0.555251896], [1.39142108], [0.42560789]], [[0.336506844], [2.69336462], [-0.081891641], [0.0233819149], [0.76918608], [0.584057629], [-0.676019549], [-0.782520652], [0.719698906], [1.33978033], [-0.279272825], [0.313035309], [-1.05213737], [0.092332691], [-0.0427948646], [1.02877712], [-0.0741262659], [0.166727409], [-0.119087324], [0.788672805], [-0.401320815], [-0.578363478], [-0.491948068], [-0.158451378], [-0.50267005], [1.0429157], [0.0735380203], [-0.897910297]], [[-1.8467648], [-1.73895586], [1.57245505], [0.43966037], [-0.0891231373], [-0.0259371046], [0.513630092], [-1.2972405], [-1.44140089], [-0.337752581], [0.974461675], [0.789644003], [-2.26118731], [-0.0507958271], [-0.644426286], [0.815869093], [0.220588699], [-0.269350529], [1.06826258], [-0.511449814], [0.599324703], [0.610500872], [-1.42896533], [-0.848673403], [1.07339847], [0.217722014], [-0.985864699], [0.448658913]], [[0.8556903], [-1.00952423], [-0.494305193], [-0.663861513], [1.074368], [0.0433590598], [0.567575157], [-0.308971077], [-0.703229308], [-2.03543067], [1.11957741], [0.67444551], [1.40356827], [-1.41711938], [-1.21145463], [-0.547020078], [0.57082361], [-0.996654093], [0.42284748], [-0.149060592], [-0.0822987109], [-0.52568543], [0.957372785], [-0.221708342], [0.825942218], [0.431177408], [-1.48766363], [-0.186143577]], [[1.185987], [1.78661466], [0.593024433], [0.802919328], [-1.5018369], [-1.16715097], [-0.695944667], [-1.70207405], [1.58165896], [-0.119928345], [-0.375099361], [0.586464524], [0.762576878], [-0.788888872], [1.11347198], [-0.207798123], [1.55673623], [-0.601261795], [0.479875028], [0.137834594], [0.687370658], [-0.389357299], [-0.797320902], [-0.674687743], [-1.19136512], [-1.16266739], [0.751781642], [-0.312029034]], [[-0.438029319], [-0.482722282], [0.509705484], [-2.29501677], [1.14759338], [-0.691866338], [-0.165856108], [-0.687538087], [-0.302764624], [-0.889853835], [-0.895094872], [-0.184980303], [-0.378466308], [-0.261140645], [-1.30532873], [-0.997758985], [-0.715995908], [-2.23052526], [-0.367753863], [0.234907091], [0.589695454], [-0.124931321], [-2.38305092], [-1.99669051], [0.649343908], [2.1673851], [0.761950493], [-0.857119083]], [[1.08631444], [0.850516915], [0.358447611], [0.152253434], [0.587350249], [0.417242736], [-0.487737715], [1.30954576], [0.168136656], [1.11370111], [0.75721091], [0.111706667], [1.12816751], [1.74772692], [0.325431943], [-0.431021631], [0.970977068], [1.67086101], [-0.918734193], [1.32473433], [0.0811438337], [-1.27158332], [-0.0777964965], [1.13624942], [0.0267926306], [-0.114960164], [0.183603972], [2.30384326]], [[0.619051397], [-0.33316946], [0.250380039], [-1.21435595], [-0.886605144], [-0.644823492], [-1.0291599], [0.437302142], [-0.62614125], [-1.64470637], [0.407202274], [0.544977009], [0.0444848016], [-0.202407882], [-1.31654048], [-0.129009724], [-1.01081645], [0.635884881], [1.43841529], [-0.574217081], [0.456742853], [0.30428946], [0.461139441], [-0.459994793], [0.703237116], [-0.679494679], [0.0924349204], [1.59983051]], [[0.695928574], [-2.43939304], [-0.588034332], [-0.266026825], [1.33245862], [0.934119225], [0.90558964], [-1.1627152], [0.386840165], [0.227862567], [-0.648688734], [-0.347289473], [-0.294617295], [-0.413749605], [-0.981053472], [0.532352507], [0.147307634], [0.981656194], [0.820834696], [-0.100734137], [0.283015341], [-0.0750106275], [1.3004477], [-0.42496717], [0.544046283], [1.16734147], [-0.283056438], [-1.27143431]], [[0.332326502], [1.25879014], [-0.185278416], [0.902772129], [0.815803885], [3.02288699], [-1.02985501], [0.275474608], [-0.363981336], [-1.71805573], [0.0211413465], [1.36740875], [1.89228582], [0.927743673], [0.732993186], [0.967047572], [-0.497091562], [1.21467102], [0.64232558], [1.08822477], [2.01905251], [-1.33466756], [-1.34349024], [-0.707021713], [-1.23846126], [0.339277714], [0.884667933], [0.475289702]], [[-0.558524549], [0.639037251], [0.318379641], [1.62989366], [-0.534379005], [-0.703389883], [0.00872256141], [2.35306859], [-1.51646888], [0.973643363], [0.821015716], [-3.13230419], [0.988020241], [-0.869053185], [0.349523604], [-0.301544905], [-0.312694907], [0.48961699], [-0.819203913], [0.189824149], [0.0212133955], [-0.808824539], [0.0989582464], [0.691875279], [-0.901261091], [0.775412619], [-0.605675101], [-0.109420218]]]], requires_grad=True)
        encoder_input_image = encoder_input_image.view(1, *img_dim)
        # encoder_input_label = torch.randint(0, label_dim, (1, ))
        encoder_input_label = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]])
        print(f"---A1---Encoder input: {encoder_input_image.shape}, {encoder_input_label.shape}")
        decoder_input_latent = torch.randn(1, latent_dim, requires_grad=True)
        # decoder_input_label = torch.randint(0, label_dim, (1, ))
        decoder_input_label = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        print(f"---A2---Decoder input: {decoder_input_latent.shape}, {decoder_input_label.shape}")
        vae_input_image = torch.randn(1, *img_dim, requires_grad=True)
        # vae_input_label = torch.randint(0, label_dim, (1, ))
        vae_input_label = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        print(f"---A3---VAE input: {vae_input_image.shape}, {vae_input_label.shape}")
        
        onnx_path = "onnx_models"
        encoder_onnx_path = os.path.join(onnx_path, f"encoder-{en_drate}-{de_drate}-big.onnx")
        print(f"---A4---Encoder onnx path: {encoder_onnx_path}")
        decoder_onnx_path = os.path.join(onnx_path, f"decoder-{en_drate}-{de_drate}-big.onnx")
        vae_onnx_path = os.path.join(onnx_path, f"vae_model-{en_drate}-{de_drate}-big.onnx")
        torch.onnx.export(encoder, 
                        (encoder_input_image, encoder_input_label),
                        encoder_onnx_path,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=['input_image', 'input_label'],
                        output_names=['output_mu', 'output_logstd'],
                        dynamic_axes={'input_image': {0: 'batch_size'},
                                        'input_label': {0: 'batch_size'},
                                        'output_mu': {0: 'batch_size'},
                                        'output_logstd': {0: 'batch_size'}})
        torch.onnx.export(decoder,
                        (decoder_input_latent, decoder_input_label),
                        decoder_onnx_path,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=['input_latent', 'input_label'],
                        output_names=['output_image'],
                        dynamic_axes={'input_latent': {0: 'batch_size'},
                                        'input_label': {0: 'batch_size'},
                                        'output_image': {0: 'batch_size'}})
        torch.onnx.export(vae_model,
                        (vae_input_image, vae_input_label),
                        vae_onnx_path,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=['input_image', 'input_label'],
                        output_names=['output_image', 'output_mu', 'output_logstd'],
                        dynamic_axes={'input_image': {0: 'batch_size'},
                                        'input_label': {0: 'batch_size'},
                                        'output_image': {0: 'batch_size'},
                                        'output_mu': {0: 'batch_size'},
                                        'output_logstd': {0: 'batch_size'}})
        print("---A5---ONNX model exported successfully.")
                        
        ### [TEST] load onnx model and compare with pytorch model. PASSED!
        # ort_session = onnxruntime.InferenceSession(encoder_onnx_path, providers=["CPUExecutionProvider"])
        # encoder_onnx_model = onnx.load(encoder_onnx_path)
        
        # def to_numpy(tensor):
        #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(encoder_input_image),
        #             ort_session.get_inputs()[1].name: to_numpy(encoder_input_label)}
        # ort_outs = ort_session.run(None, ort_inputs)
        # torch_out = encoder(encoder_input_image, encoder_input_label)
        # print(f"---A6---Encoder ONNX output: \n {ort_outs[0]} \n {ort_outs[1]}")
        # print(f"---A7---Encoder PyTorch output: \n {torch_out[0]} \n {torch_out[1]}")
        ### [TEST END]
        
        # summary(encoder, [(1, *img_dim), (1, 11)], device="cpu")