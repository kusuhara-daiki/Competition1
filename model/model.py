
from a_model import Model

def choose_model(cfg, args, device): 
    if cfg.model_name=='a_model':   
        model = Model()
    elif cfg.model_name == 'b_model':
          model = Model()

    if cfg.switch.model.learned_model == True:
        model = torch.load(args.after_model_path)
    else:
        model = Model().to(device)
    return model